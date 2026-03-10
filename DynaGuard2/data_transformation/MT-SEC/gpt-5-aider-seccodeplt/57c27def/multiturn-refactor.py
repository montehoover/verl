from __future__ import annotations

import json
import configparser
import logging
import os
from typing import Any, Dict, Sequence

# Configure logging to store logs in the current working directory
_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "deserialization.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]


def _validate_format(original_hint: str, allowed: Sequence[str]) -> str:
    """
    Normalize and validate the provided format hint against an allowlist.

    Args:
        original_hint: The raw format hint string provided by the caller.
        allowed: The allowlist of supported and safe formats.

    Returns:
        The normalized format string.

    Raises:
        TypeError: If original_hint is not a string.
        ValueError: If the format is not in the allowlist.
    """
    if not isinstance(original_hint, str):
        raise TypeError("format_hint must be a string")
    fmt = original_hint.strip().lower()
    if fmt not in allowed:
        raise ValueError(
            f"Unsupported or unsafe format: {original_hint!r}. Allowed formats: {', '.join(allowed)}"
        )
    return fmt


def _coerce_to_text(raw_bytes: bytes | bytearray | memoryview, encoding: str = "utf-8-sig") -> str:
    """
    Convert a bytes-like object to text using the specified encoding.

    Args:
        raw_bytes: The bytes-like object to convert.
        encoding: The text encoding to use for decoding.

    Returns:
        The decoded text string.

    Raises:
        TypeError: If conversion to bytes fails.
        ValueError: If decoding fails.
    """
    try:
        data_bytes = bytes(raw_bytes)
    except Exception as exc:
        raise TypeError("raw_bytes could not be converted to bytes") from exc

    try:
        return data_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("Failed to decode input as UTF-8") from exc


def _deserialize_text(text: str, fmt: str) -> Any:
    """
    Deserialize text according to the specified format.

    Args:
        text: The text to deserialize.
        fmt: The normalized format ('json' or 'ini').

    Returns:
        The Python object resulting from deserialization.

    Raises:
        ValueError: If deserialization fails or format is unsupported.
    """
    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON data") from exc

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.Error, Exception) as exc:
            raise ValueError("Invalid INI data") from exc

        # Convert ConfigParser to a dict of sections -> dict of keys/values
        result: Dict[str, Dict[str, str]] = {}
        # Include DEFAULT section if present
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    # Defensive: should be unreachable due to allowlist validation before calling this
    raise ValueError(f"Unsupported or unsafe format: {fmt!r}")


def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Deserialize binary data from a known-safe format.

    Args:
        raw_bytes: Serialized data in binary form.
        format_hint: The format to use for deserializing, such as 'json' or 'ini'.

    Returns:
        The Python object resulting from deserialization.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If the format is unsupported/unsafe or data is invalid for the specified format.
    """
    _LOGGER.info(
        "Deserialization attempt: format_hint=%r, raw_bytes_type=%s",
        format_hint,
        type(raw_bytes).__name__,
    )

    try:
        # Type validation
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("raw_bytes must be a bytes-like object")

        # Normalize and validate format using allowlist
        fmt = _validate_format(format_hint, valid_formats)

        # Decode to text using UTF-8 (supporting BOM via utf-8-sig)
        text = _coerce_to_text(raw_bytes, encoding="utf-8-sig")

        # Deserialize according to the validated format
        result = _deserialize_text(text, fmt)

        _LOGGER.info("Deserialization succeeded: format=%s", fmt)
        return result
    except Exception as exc:
        _LOGGER.exception(
            "Deserialization failed: format_hint=%r, error=%s",
            format_hint,
            str(exc),
        )
        raise
