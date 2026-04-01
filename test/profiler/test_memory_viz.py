# Owner(s): ["oncall: profiler"]

import os
import shutil
import subprocess
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase


class TestMemoryViz(TestCase):
    def test_process_alloc_data(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node.js not available")
        test_js = os.path.join(os.path.dirname(__file__), "test_memory_viz.js")
        result = subprocess.run(
            [node, test_js],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            self.fail(f"JS tests failed:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    run_tests()
