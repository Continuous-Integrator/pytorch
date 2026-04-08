# Owner(s): ["module: cuda graphs"]

"""Tests for CUDA graph kernel annotation via mark_kernels."""

import unittest

import torch
from torch.cuda._graph_annotations import (
    _is_tools_id_unavailable,
    clear_kernel_annotations,
    enable_annotations,
    get_kernel_annotations,
    mark_kernels,
    remap_to_exec_graph,
    resolve_pending_annotations,
)
from torch.testing._internal.common_utils import run_tests, TestCase


TEST_CUDA = torch.cuda.is_available()

try:
    import cuda.bindings.runtime  # noqa: F401

    TEST_CUDA_BINDINGS = True
except ImportError:
    TEST_CUDA_BINDINGS = False


@unittest.skipUnless(TEST_CUDA, "CUDA not available")
@unittest.skipUnless(TEST_CUDA_BINDINGS, "cuda.bindings not available")
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestMarkKernels(TestCase):
    def setUp(self):
        enable_annotations()
        clear_kernel_annotations()

    def tearDown(self):
        clear_kernel_annotations()

    def test_noop_outside_capture(self):
        x = torch.randn(8, device="cuda")
        with mark_kernels("test"):
            _ = x + 1
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_single_scope(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("phase_a"):
                _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, "phase_a")

    def test_multiple_scopes_no_overlap(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("scope_1"):
                _ = x + 1
            with mark_kernels("scope_2"):
                _ = x * 2
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        scope_1_ids = set()
        scope_2_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(len(anns), 1)
            if anns[0] == "scope_1":
                scope_1_ids.add(tid)
            elif anns[0] == "scope_2":
                scope_2_ids.add(tid)

        self.assertGreater(len(scope_1_ids), 0)
        self.assertGreater(len(scope_2_ids), 0)
        self.assertEqual(len(scope_1_ids & scope_2_ids), 0)

    def test_dict_annotation(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        annotation = {"name": "all_gather", "Group size": 2, "dtype": "bfloat16"}
        with torch.cuda.graph(graph):
            with mark_kernels(annotation):
                _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(anns[0]["name"], "all_gather")
            self.assertEqual(anns[0]["Group size"], 2)

    def test_clear_resets_state(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("test"):
                _ = x + 1
            resolve_pending_annotations()

        self.assertGreater(len(get_kernel_annotations()), 0)
        clear_kernel_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_resolve_without_scopes_is_noop(self):
        resolve_pending_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_scope_with_no_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            _ = x + 1
            with mark_kernels("empty"):
                pass
            _ = x * 2
            resolve_pending_annotations()

        for anns in get_kernel_annotations().values():
            for ann in anns:
                self.assertNotEqual(ann, "empty")

    def test_only_annotates_scope_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            _ = x + 1
            _ = x * 2
            with mark_kernels("tagged"):
                _ = x + 3
            _ = x - 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        total_annotated = sum(len(anns) for anns in annotations.values())
        self.assertGreater(total_annotated, 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, "tagged")

    def test_nested_scopes_innermost_wins(self):
        """With nested string scopes, the innermost name wins."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("outer"):
                _ = x + 1  # outer only
                with mark_kernels("inner"):
                    _ = x * 2  # nested: inner should win
                _ = x - 1  # outer only
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        outer_ids = set()
        inner_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(
                len(anns), 1, f"toolsId {hex(tid)} has {len(anns)} annotations"
            )
            ann = anns[0]
            # Outer-only kernels stay as raw strings
            if ann == "outer":
                outer_ids.add(tid)
            # Nested kernels get merged into a dict
            elif isinstance(ann, dict) and ann["name"] == "inner":
                inner_ids.add(tid)

        self.assertGreater(len(outer_ids), 0, "Should have outer-only kernels")
        self.assertGreater(len(inner_ids), 0, "Should have inner kernels")
        self.assertEqual(len(outer_ids & inner_ids), 0)

    def test_nested_dict_scopes_inner_wins_common_keys(self):
        """With truly nested dict scopes, inner wins for common keys,
        outer-only keys are preserved."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        outer_ann = {"name": "ag_collective", "stream": 71}
        inner_ann = {
            "name": "all_gather",
            "stream": 62,
            "In msg nelems": 1024,
            "dtype": "bfloat16",
        }

        with torch.cuda.graph(graph):
            with mark_kernels(outer_ann):
                _ = x + 1  # outer only
                with mark_kernels(inner_ann):
                    _ = x * 2  # nested
                _ = x - 1  # outer only
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        outer_only_ids = set()
        nested_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertIsInstance(ann, dict)
            if ann["name"] == "ag_collective":
                outer_only_ids.add(tid)
            elif ann["name"] == "all_gather":
                nested_ids.add(tid)
                # Inner wins for common keys
                self.assertEqual(ann["stream"], 62)
                # Inner-only keys preserved
                self.assertEqual(ann["In msg nelems"], 1024)
                self.assertEqual(ann["dtype"], "bfloat16")

        self.assertGreater(len(outer_only_ids), 0, "Should have outer-only kernels")
        self.assertGreater(len(nested_ids), 0, "Should have nested kernels")

    def test_same_range_scopes_inner_wins_common_keys(self):
        """With same-range scopes (inner ctx exits first), inner wins
        for common keys, outer-only keys are preserved."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        outer_ann = {"name": "ag_collective", "stream": 71}
        inner_ann = {
            "name": "all_gather",
            "stream": 62,
            "In msg nelems": 1024,
            "dtype": "bfloat16",
        }

        with torch.cuda.graph(graph):
            # Both scopes wrap the same kernels; inner exits first.
            with mark_kernels(outer_ann):
                with mark_kernels(inner_ann):
                    _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertIsInstance(ann, dict)
            # Inner wins for common keys
            self.assertEqual(ann["name"], "all_gather", "Inner name should win")
            self.assertEqual(ann["stream"], 62, "Inner stream should win")
            # Inner-only keys preserved
            self.assertEqual(ann["In msg nelems"], 1024)
            self.assertEqual(ann["dtype"], "bfloat16")

    def test_remap_to_exec_graph(self):
        from cuda.bindings import runtime as cuda_runtime

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("test"):
                _ = x + 1
            resolve_pending_annotations()

        annotations_before = dict(get_kernel_annotations())
        self.assertGreater(len(annotations_before), 0)

        exec_handle = cuda_runtime.cudaGraphExec_t(
            init_value=graph.raw_cuda_graph_exec()
        )
        _, exec_graph_id = cuda_runtime.cudaGraphExecGetId(exec_handle)

        remap_to_exec_graph(graph)

        annotations_after = get_kernel_annotations()
        self.assertEqual(len(annotations_after), len(annotations_before))
        for tools_id in annotations_after:
            self.assertEqual(tools_id >> 32, exec_graph_id)

    def test_disabled_is_noop(self):
        from torch.cuda._graph_annotations import disable_annotations

        disable_annotations()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("should_not_appear"):
                _ = x + 1
            resolve_pending_annotations()

        self.assertEqual(len(get_kernel_annotations()), 0)

        # Re-enable for other tests
        enable_annotations()


class TestAnnotateTrace(TestCase):
    """Tests for the trace post-processing functions (no GPU required)."""

    def _make_kernel_event(
        self, name: str, tid: int, ts: float, dur: float, graph_node_id: int = 0
    ) -> dict:
        return {
            "cat": "kernel",
            "name": name,
            "tid": tid,
            "ts": ts,
            "dur": dur,
            "args": {"graph node id": graph_node_id},
        }

    def test_annotate_dict_annotation(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern_a", 1, 100, 10, graph_node_id=42),
                self._make_kernel_event("kern_b", 1, 200, 10, graph_node_id=0),
            ]
        }
        annotations = {42: [{"name": "all_gather", "dtype": "bf16"}]}
        count = annotate_trace(trace, annotations, default_stream=7)

        self.assertEqual(count, 1)
        args = trace["traceEvents"][0]["args"]
        self.assertEqual(args["name"], "all_gather")
        self.assertEqual(args["dtype"], "bf16")
        # Graphed event should be moved to default stream
        self.assertEqual(trace["traceEvents"][0]["tid"], 7)

    def test_annotate_string_annotation(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern", 1, 100, 10, graph_node_id=99),
            ]
        }
        annotations = {99: ["phase_a"]}
        count = annotate_trace(trace, annotations, default_stream=7)

        self.assertEqual(count, 1)
        self.assertEqual(trace["traceEvents"][0]["args"]["annotation"], "phase_a")

    def test_annotate_unannotated_graphed_events_get_default_stream(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern", 5, 100, 10, graph_node_id=50),
            ]
        }
        # No annotation for graph_node_id=50
        count = annotate_trace(trace, {}, default_stream=7)

        self.assertEqual(count, 0)
        self.assertEqual(trace["traceEvents"][0]["tid"], 7)

    def test_annotate_non_graphed_events_untouched(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        event = self._make_kernel_event("kern", 3, 100, 10, graph_node_id=0)
        trace = {"traceEvents": [event]}
        annotate_trace(trace, {}, default_stream=7)

        # graph_node_id=0 means not graphed, tid should be unchanged
        self.assertEqual(trace["traceEvents"][0]["tid"], 3)

    def test_annotate_stream_from_annotation(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern", 1, 100, 10, graph_node_id=42),
            ]
        }
        annotations = {42: [{"name": "rs", "stream": 62}]}
        annotate_trace(trace, annotations, default_stream=7)

        self.assertEqual(trace["traceEvents"][0]["tid"], 62)

    def test_move_overlapping_to_stream(self):
        from torch.cuda._annotate_cuda_graph_trace import _move_overlapping_to_stream

        trace = {
            "traceEvents": [
                self._make_kernel_event("a", 7, 100, 20, graph_node_id=1),
                # Overlaps: starts at 110 < 120 (end of a)
                self._make_kernel_event("b", 7, 110, 15, graph_node_id=2),
                # No overlap: starts at 130 >= 125 (end of b on stream 8)
                self._make_kernel_event("c", 7, 130, 10, graph_node_id=3),
            ]
        }
        moved = _move_overlapping_to_stream(trace, default_stream=7, overlap_stream=8)

        self.assertEqual(moved, 1)
        self.assertEqual(trace["traceEvents"][0]["tid"], 7)
        self.assertEqual(trace["traceEvents"][1]["tid"], 8)
        self.assertEqual(trace["traceEvents"][2]["tid"], 7)

    def test_fix_overlapping_timestamps(self):
        from torch.cuda._annotate_cuda_graph_trace import _fix_overlapping_timestamps

        trace = {
            "traceEvents": [
                self._make_kernel_event("a", 7, 100, 10, graph_node_id=1),
                # Overlaps by 0.5us (within threshold)
                self._make_kernel_event("b", 7, 109.5, 10, graph_node_id=2),
            ]
        }
        adjusted = _fix_overlapping_timestamps(trace, max_adjust_us=1.0)

        self.assertEqual(adjusted, 1)
        self.assertEqual(trace["traceEvents"][1]["ts"], 110)

    def test_fix_overlapping_timestamps_large_overlap_skipped(self):
        from torch.cuda._annotate_cuda_graph_trace import _fix_overlapping_timestamps

        trace = {
            "traceEvents": [
                self._make_kernel_event("a", 7, 100, 10, graph_node_id=1),
                # Overlaps by 5us (exceeds threshold)
                self._make_kernel_event("b", 7, 105, 10, graph_node_id=2),
            ]
        }
        adjusted = _fix_overlapping_timestamps(trace, max_adjust_us=1.0)

        self.assertEqual(adjusted, 0)
        # Should be unchanged
        self.assertEqual(trace["traceEvents"][1]["ts"], 105)

    def test_noise_removal(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern", 7, 100, 10, graph_node_id=1),
                # Noise: gpu_user_annotation on stream 99 with no work
                {
                    "cat": "gpu_user_annotation",
                    "tid": 99,
                    "ts": 100,
                    "dur": 10,
                },
                # ac2g flow finish on stream 99 (no work there)
                {"cat": "ac2g", "ph": "f", "tid": 99, "ts": 100},
            ]
        }
        annotate_trace(trace, {}, default_stream=7)

        # Noise events on stream 99 should be removed
        cats = [e.get("cat") for e in trace["traceEvents"]]
        self.assertNotIn("gpu_user_annotation", cats)
        tids = [e.get("tid") for e in trace["traceEvents"] if e.get("cat") == "ac2g"]
        for tid in tids:
            self.assertNotEqual(tid, 99)

    def test_ac2g_flow_events_follow_kernel(self):
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace

        trace = {
            "traceEvents": [
                self._make_kernel_event("kern", 5, 100, 10, graph_node_id=42),
                {"cat": "ac2g", "ph": "f", "tid": 5, "ts": 100},
            ]
        }
        annotations = {42: [{"name": "ag", "stream": 62}]}
        annotate_trace(trace, annotations, default_stream=7)

        # Both kernel and ac2g should be moved to stream 62
        self.assertEqual(trace["traceEvents"][0]["tid"], 62)
        self.assertEqual(trace["traceEvents"][1]["tid"], 62)


if __name__ == "__main__":
    run_tests()
