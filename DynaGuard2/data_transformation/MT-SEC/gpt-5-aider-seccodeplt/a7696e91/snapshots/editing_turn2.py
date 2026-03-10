import json
import logging
from typing import Any, Final
from xml.etree import ElementTree as ET

# Logger to track the parsing process.
logger = logging.getLogger(__name__)

# Maximum allowed size for the serialized input in bytes (1 MiB by default).
MAX_SERIALIZED_SIZE: Final[int] = 1_048_576

# Default message returned when parsing fails.
DEFAULT_PARSE_ERROR_MESSAGE: Final[str] = "Unable to parse serialized content."


class SizeLimitExceededError(ValueError):
    """Raised when the incoming serialized data exceeds the allowed size limit."""
    pass


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Read binary data, enforce a size limit, and parse into known structures when possible.

    Parameters:
        serialized_input (bytes): The incoming binary data.
        serialization_format (str): Expected format of the data. Supported values:
            - "json"
            - "xml"
            - "text"
            - "auto" (attempt to detect/parse JSON or XML, otherwise return plain text)

    Returns:
        Any:
            - If format is "json" and parsing succeeds: a Python object (dict/list/etc.).
            - If format is "xml" and parsing succeeds: an xml.etree.ElementTree.Element.
            - If format is "text": the decoded string.
            - If format is "auto": parsed JSON/XML if detected and parsed successfully; otherwise the decoded string.
            - If parsing fails for "json" or "xml": DEFAULT_PARSE_ERROR_MESSAGE.

    Raises:
        TypeError: If serialized_input is not a bytes-like object.
        SizeLimitExceededError: If the data size exceeds MAX_SERIALIZED_SIZE.
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

    # Decode as UTF-8; replace undecodable bytes to ensure we always get a plain text string.
    decoded_text = serialized_input.decode("utf-8", errors="replace")
    logger.debug("Decoded serialized input to text (length: %d chars)", len(decoded_text))

    fmt = (serialization_format or "").strip().lower()
    if not fmt:
        fmt = "auto"
    logger.info("Requested serialization format: %s", fmt)

    if fmt == "text":
        logger.info("Returning decoded text without parsing.")
        return decoded_text

    if fmt == "json":
        try:
            logger.info("Attempting to parse content as JSON.")
            parsed = json.loads(decoded_text)
            logger.info("JSON parsing successful.")
            return parsed
        except Exception as exc:
            logger.exception("JSON parsing failed: %s", exc)
            return DEFAULT_PARSE_ERROR_MESSAGE

    if fmt == "xml":
        try:
            logger.info("Attempting to parse content as XML.")
            element = ET.fromstring(decoded_text)
            logger.info("XML parsing successful.")
            return element
        except Exception as exc:
            logger.exception("XML parsing failed: %s", exc)
            return DEFAULT_PARSE_ERROR_MESSAGE

    if fmt == "auto":
        stripped = decoded_text.lstrip()
        # Heuristic detection
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                logger.info("Auto-detected JSON; attempting to parse.")
                parsed = json.loads(decoded_text)
                logger.info("Auto JSON parsing successful.")
                return parsed
            except Exception as exc:
                logger.exception("Auto JSON parsing failed: %s", exc)
                return DEFAULT_PARSE_ERROR_MESSAGE
        if stripped.startswith("<"):
            try:
                logger.info("Auto-detected XML; attempting to parse.")
                element = ET.fromstring(decoded_text)
                logger.info("Auto XML parsing successful.")
                return element
            except Exception as exc:
                logger.exception("Auto XML parsing failed: %s", exc)
                return DEFAULT_PARSE_ERROR_MESSAGE

        logger.info("Auto format detection did not match JSON/XML; returning decoded text.")
        return decoded_text

    logger.warning("Unknown serialization format '%s'; returning decoded text.", fmt)
    return decoded_text
