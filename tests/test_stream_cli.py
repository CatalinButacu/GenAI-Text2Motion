"""Epic A-2: CLI smoke tests for --mode stream.

These tests verify that main() routes correctly and prints expected output
without requiring checkpoints or real motion generation.
"""
from __future__ import annotations

import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from main import main
from src.modules.runtime import StreamCapabilityError


def runMain(argv: list[str]) -> tuple[int, str]:
    """Run main() with the given argv; capture stdout; return (exit_code, output)."""
    out = StringIO()

    with patch("sys.argv", ["main.py"] + argv):
        with patch("sys.stdout", out):
            try:
                main()
                code = 0
            except SystemExit as e:  # NOSONAR – intentionally capturing exit code in test helper
                code = e.code if isinstance(e.code, int) else 1

    return code, out.getvalue()


SUMMARY_STUB = {
    "produced_chunks": 3,
    "consumed_chunks": 3,
    "dropped_chunks": 0,
    "bad_chunks": 0,
    "first_chunk_latency_ms": 12.5,
    "inter_chunk_p50_ms": 8.0,
    "inter_chunk_p95_ms": 9.1,
    "wall_time_ms": 42.0,
}


class TestStreamCLI(unittest.TestCase):

    def test_stream_mode_prints_summary_keys(self):
        with patch("src.pipeline.Pipeline.validate_streaming", return_value=SUMMARY_STUB):
            code, output = runMain(["a person walks", "--mode", "stream"])

        self.assertEqual(code, 0)

        for key in SUMMARY_STUB:
            self.assertIn(key, output)

    def test_stream_mode_prints_summary_header(self):
        with patch("src.pipeline.Pipeline.validate_streaming", return_value=SUMMARY_STUB):
            code, output = runMain(["a person walks", "--mode", "stream"])

        self.assertEqual(code, 0)
        self.assertIn("stream summary", output)

    def test_max_stream_chunks_arg_forwarded(self):
        capturedKwargs = {}

        def captureCall(prompt, **kwargs):
            capturedKwargs.update(kwargs)
            return SUMMARY_STUB

        with patch("src.pipeline.Pipeline.validate_streaming", side_effect=captureCall):
            runMain(["a person walks", "--mode", "stream", "--max-stream-chunks", "5"])

        self.assertEqual(capturedKwargs.get("max_stream_chunks"), 5)

    def test_capability_error_prints_clean_message(self):
        errMsg = "checkpoint was trained with bidirectional=True"

        with patch(
            "src.pipeline.Pipeline.validate_streaming",
            side_effect=StreamCapabilityError(errMsg),
        ):
            code, output = runMain(["a person walks", "--mode", "stream"])

        self.assertEqual(code, 0)
        self.assertIn("error:", output)
        self.assertIn(errMsg, output)

    def test_file_mode_does_not_call_validate_streaming(self):
        with patch("src.pipeline.Pipeline.validate_streaming") as mockValidate:
            with patch("src.pipeline.Pipeline.render_to_file", return_value=None):
                runMain(["a person walks", "--mode", "file"])

        mockValidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
