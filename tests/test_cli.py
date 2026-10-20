import argparse
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from ollamamaker.cli import _build_parser, _configured_parallel
from ollamamaker.models import OllamaError


class ParserTests(unittest.TestCase):
    def test_profile_and_context_are_mutually_exclusive(self) -> None:
        parser = _build_parser()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["model", "--profile", "long", "--context", "98304"])

    def test_numeric_tuning_options_must_be_positive(self) -> None:
        parser = _build_parser()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["model", "--output", "0"])


class ParallelConfigurationTests(unittest.TestCase):
    def test_environment_parallelism_is_used_by_default(self) -> None:
        args = argparse.Namespace(parallel=None)

        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "3"}):
            self.assertEqual(_configured_parallel(args), 3)

    def test_invalid_environment_parallelism_is_rejected(self) -> None:
        args = argparse.Namespace(parallel=None)

        with (
            patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "many"}),
            self.assertRaises(OllamaError),
        ):
            _configured_parallel(args)


if __name__ == "__main__":
    unittest.main()
