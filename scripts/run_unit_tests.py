#!/usr/bin/env python3
"""Run the full unit test suite. Equivalent to `pytest tests/ -q`."""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["tests/", "-q"]))
