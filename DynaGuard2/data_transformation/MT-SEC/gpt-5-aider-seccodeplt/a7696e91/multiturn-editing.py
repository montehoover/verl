import json
import logging
import configparser
from typing import Any, Final, Dict
from xml.etree import ElementTree as ET  # Kept if other modules import; not used here.

# Logger to track the parsing process.
logger = logging.getLogger(__name__)

# Maximum allowed size for the serialized input in bytes (1 MiB by default).
MAX_SERIALIZED_SIZE: Final[int] = 1_048_576

# Approved, secure formats for deserialization.
approved_formats = ["json", "ini"]


class SizeLimitExceededError(ValueError):
    """Raised when the incoming serialized data exceeds the allowed size limit."""
    pass


class UnsafeFormatError(ValueError):
    """Raised when an unsafe or unapproved serialization format is provided."""
    pass


class ParseError(ValueError):
    """Raised when parsing fails for an approved format."""
    pass


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Securely interpret binary data, limited to approved formats only.

    Parameters:
        serialized_input (bytes): The incoming binary data.
        serialization_format (str): Expected format of the data. Must be one of:
            - "json"
            - "ini"

    Returns:
        Any:
            - For "json": a Python object (dict/list/etc.).
            - For "ini": a dict of sections to key/value mappings. If defaults exist, they are
              returned under the "DEFAULT" key.

    Raises:
        TypeError: If serialized_input is not a bytes-like object.
        SizeLimitExceededError: If the data size exceeds MAX_SERIALIZED_SIZE.
        UnsafeFormatError: If an unapproved serialization format is provided.
        ParseError: If decoding or parsing fails.
    """
    if not isinstance(serialized_input, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_input must be a bytes-like object")

    size = len(serialized_input)
    logger.debug("Received serialized input: %d bytes", size)
    if size > MAX_SERIALIZED_SIZE:
        logger.error(
            "Serialized content size %d bytes exceeds limit of %d bytes.",
            size,
            MAX_SERIALIZED_SIZE,
        )
        raise SizeLimitExceededError(
            f"Serialized content size {size} bytes exceeds limit of {MAX_SERIALIZED_SIZE} bytes."
        )

    # Normalize to bytes
    if isinstance(serialized_input, memoryview):
        serialized_input = serialized_input.tobytes()
    elif isinstance(serialized_input, bytearray):
        serialized_input = bytes(serialized_input)

    fmt = (serialization_format or "").strip().lower()
    logger.info("Requested serialization format: %s", fmt)
    if fmt not in approved_formats:
        logger.error("Unsafe or unapproved format requested: %s", fmt)
        raise UnsafeFormatError(f"Unapproved serialization format: {fmt!r}")

    # Decode as UTF-8 strictly; fail fast on invalid input.
    try:
        decoded_text = serialized_input.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        logger.exception("UTF-8 decoding failed.")
        raise ParseError("Failed to decode input as UTF-8.") from exc

    logger.debug("Decoded serialized input to text (length: %d chars)", len(decoded_text))

    if fmt == "json":
        try:
            logger.info("Attempting to parse content as JSON.")
            parsed = json.loads(decoded_text)
            logger.info("JSON parsing successful.")
            return parsed
        except json.JSONDecodeError as exc:
            logger.exception("JSON parsing failed.")
            raise ParseError("JSON parsing failed.") from exc

    if fmt == "ini":
        try:
            logger.info("Attempting to parse content as INI.")
            parser = configparser.ConfigParser(interpolation=None, strict=True)
            # Preserve option case; avoid implicit lowercasing
            parser.optionxform = str  # type: ignore[attr-defined]
            parser.read_string(decoded_text)

            result: Dict[str, Dict[str, str]] = {}
            defaults = parser.defaults()
            if defaults:
                result["DEFAULT"] = dict(defaults)
            for section in parser.sections():
                # raw=True to avoid any interpolation
                result[section] = {k: v for k, v in parser.items(section, raw=True)}
            logger.info("INI parsing successful.")
            return result
        except (configparser.Error, Exception) as exc:
            logger.exception("INI parsing failed.")
            raise ParseError("INI parsing failed.") from exc

    # This point should be unreachable due to the approved_formats check.
    logger.error("Reached unreachable code path for format: %s", fmt)
    raise UnsafeFormatError(f"Unapproved serialization format: {fmt!r}")
