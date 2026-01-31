"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys


def _preload_numpy_without_macos_check() -> None:
    """Preload NumPy while bypassing the macOS sanity check.

    This avoids a hard crash seen with some macOS BLAS/LAPACK builds during
    NumPy's import-time polyfit check.
    """
    if sys.platform != "darwin":
        return

    original_platform = sys.platform
    try:
        sys.platform = "linux"
        import numpy  # noqa: F401
    finally:
        sys.platform = original_platform


_preload_numpy_without_macos_check()


