"""
melage.api._progress
====================
Headless replacements for the PyQt5 progress-bar / label widgets that MELAGE
plugin internals use to report progress.

Usage
-----
    bar = PrintProgress("BET")            # prints percentage to stdout
    bar = SilentProgress()                # completely silent
    bar = CallbackProgress(my_callback)   # calls my_callback(pct, msg)
"""

from __future__ import annotations
from typing import Callable, Optional


class _BaseProgress:
    """Shared interface expected by all MELAGE plugin internals."""

    def setValue(self, value: int) -> None:
        pass

    def setText(self, text: str) -> None:
        pass

    def setVisible(self, visible: bool) -> None:
        pass

    def isVisible(self) -> bool:
        return True

    # Some plugins call .value() to read the current percentage
    def value(self) -> int:
        return self._value

    _value: int = 0


class SilentProgress(_BaseProgress):
    """Drop-in PyQt progress bar that does absolutely nothing."""

    def setValue(self, value: int) -> None:
        self._value = value


class PrintProgress(_BaseProgress):
    """
    Prints progress to stdout.

    Parameters
    ----------
    label : str
        Prefix shown in the progress line, e.g. "BET" → "[BET] 42%".
    """

    def __init__(self, label: str = "") -> None:
        self._label = label
        self._value = 0
        self._last_msg = ""

    def setValue(self, value: int) -> None:
        self._value = int(value)
        prefix = f"[{self._label}] " if self._label else ""
        msg = self._last_msg
        suffix = f" — {msg}" if msg else ""
        print(f"\r{prefix}{self._value:3d}%{suffix}", end="", flush=True)
        if self._value >= 100:
            print()  # newline on completion

    def setText(self, text: str) -> None:
        self._last_msg = text


class CallbackProgress(_BaseProgress):
    """
    Fires a user-supplied callback ``fn(pct: int, msg: str)`` on each update.

    Parameters
    ----------
    fn : callable
        Receives ``(percent: int, message: str)``.
    """

    def __init__(self, fn: Callable[[int, str], None]) -> None:
        self._fn = fn
        self._value = 0
        self._last_msg = ""

    def setValue(self, value: int) -> None:
        self._value = int(value)
        self._fn(self._value, self._last_msg)

    def setText(self, text: str) -> None:
        self._last_msg = text
        self._fn(self._value, text)


def make_progress(progress_arg, label: str = "") -> _BaseProgress:
    """
    Resolve the *progress* argument accepted by every public API function.

    Parameters
    ----------
    progress_arg :
        • ``True``  → PrintProgress (stdout)
        • ``False`` → SilentProgress
        • ``None``  → PrintProgress (same as True)
        • callable  → CallbackProgress wrapping the callable
        • already a _BaseProgress subclass → passed through unchanged
    label : str
        Label shown when printing.
    """
    if isinstance(progress_arg, _BaseProgress):
        return progress_arg
    if callable(progress_arg):
        return CallbackProgress(progress_arg)
    if progress_arg is False:
        return SilentProgress()
    return PrintProgress(label)
