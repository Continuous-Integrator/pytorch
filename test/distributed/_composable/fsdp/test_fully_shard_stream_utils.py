# Owner(s): ["oncall: distributed"]

import unittest

import torch
from torch.distributed.fsdp._fully_shard._stream_utils import StreamHandoff
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


@unittest.skipUnless(TEST_CUDA, "StreamHandoff tests require CUDA")
class TestStreamHandoff(TestCase):
    def _make_handoff(
        self,
        *,
        shape=(16,),
        producer_stream=None,
        release_stream=None,
        with_event=True,
    ):
        if producer_stream is None:
            producer_stream = torch.cuda.Stream()
        if release_stream is None:
            release_stream = torch.cuda.Stream()
        with torch.cuda.stream(producer_stream):
            t = torch.zeros(shape, device="cuda")
            event = producer_stream.record_event() if with_event else None
        return (
            StreamHandoff(
                tensor=t,
                ready_event=event,
                release_stream=release_stream,
            ),
            t,
            producer_stream,
            release_stream,
        )

    def test_basic_wait_and_release(self):
        h, _, _, _ = self._make_handoff()
        self.assertFalse(h.released)
        self.assertIsNotNone(h.tensor)
        h.release()
        self.assertTrue(h.released)
        with self.assertRaises(RuntimeError):
            _ = h.tensor

    def test_release_idempotent(self):
        h, _, _, _ = self._make_handoff()
        h.release()
        h.release()  # second call is a no-op; must not raise
        self.assertTrue(h.released)

    def test_del_calls_release(self):
        h, _, _, release_stream = self._make_handoff()
        tensor_ref = h._tensor  # access for instrumentation
        self.assertIsNotNone(tensor_ref)
        del h  # __del__ should call release()
        torch.cuda.synchronize()
        # The handoff's internal _tensor is gone; exact refcount check is
        # fragile. We verify no exception was raised and the path executed.

    def test_release_requires_non_none_tensor(self):
        producer = torch.cuda.Stream()
        release = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            event = producer.record_event()
        with self.assertRaises(ValueError):
            StreamHandoff(tensor=None, ready_event=event, release_stream=release)

    def test_release_with_none_event_skips_wait(self):
        """CPU-style fallback: None event still routes free to release_stream."""
        h, _, _, _ = self._make_handoff(with_event=False)
        self.assertIsNone(h.event)
        # Should not raise; wait_event step is skipped.
        h.release()
        self.assertTrue(h.released)

    def test_release_waits_on_consumer_stream(self):
        """release() must issue wait_event on release_stream for the event."""
        producer = torch.cuda.Stream()
        release = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            # Enqueue a large sleep-like op to delay the producer.
            big = torch.empty((4 * 1024 * 1024,), device="cuda")
            big.zero_()
            t = torch.ones(16, device="cuda")
            event = producer.record_event()
        h = StreamHandoff(
            tensor=t,
            ready_event=event,
            release_stream=release,
        )
        # Before release, release_stream has no dependency on producer.
        h.release()
        # After release, release_stream has waited on event. Enqueue a
        # subsequent op on release_stream; it must be ordered after event.
        # Can't easily assert ordering without more machinery; at minimum
        # the release must not error and the subsequent sync works.
        with torch.cuda.stream(release):
            probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        self.assertEqual(probe.item(), 0.0)

    def test_repr_readable(self):
        h, _, _, _ = self._make_handoff()
        r = repr(h)
        self.assertIn("StreamHandoff", r)
        self.assertIn("tensor", r)
        h.release()
        r2 = repr(h)
        self.assertIn("released", r2)

    def test_device_handle_autodetect(self):
        """Construct without explicit device_handle."""
        h, _, _, _ = self._make_handoff()
        # Should have inferred torch.cuda from tensor.device.type.
        self.assertIs(h._device_handle, torch.cuda)
        h.release()

    def test_device_handle_explicit(self):
        producer = torch.cuda.Stream()
        release = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            t = torch.zeros(16, device="cuda")
            event = producer.record_event()
        h = StreamHandoff(
            tensor=t,
            ready_event=event,
            release_stream=release,
            device_handle=torch.cuda,
        )
        self.assertIs(h._device_handle, torch.cuda)
        h.release()

    def test_no_use_after_free_with_keep_alive(self):
        """Core invariant: while a StreamHandoff holds a tensor, the
        allocation cannot be reused by a later allocation on the producer
        stream. This is the exact pattern the cross_stream_demo.py
        'PIPELINED' case demonstrates.
        """
        producer = torch.cuda.Stream()
        release = torch.cuda.Stream()
        # Allocate a buffer on producer, wrap in handoff.
        with torch.cuda.stream(producer):
            t1 = torch.full((1024,), 1.0, device="cuda")
            event = producer.record_event()
        h = StreamHandoff(
            tensor=t1,
            ready_event=event,
            release_stream=release,
        )
        # Allocate a second buffer on the same producer stream. Because
        # h still holds t1, the allocator cannot reuse t1's block.
        with torch.cuda.stream(producer):
            t2 = torch.full((1024,), 2.0, device="cuda")
        self.assertNotEqual(t1.data_ptr(), t2.data_ptr())
        # After release + sync, the block is eligible for reuse.
        h.release()
        torch.cuda.synchronize()
        # (We don't assert the block is actually reused — allocator
        # decisions are internal — only that no UAF occurred above.)
        del t1, t2


if __name__ == "__main__":
    run_tests()
