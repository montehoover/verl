import json
import configparser
import logging
from typing import Any

# Secure set of accepted formats designated to ensure safe deserialization practices
approved_formats = ["json", "ini"]

# Module-level logger for auditing/debugging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Deserialize binary data from potentially untrusted sources using only approved formats.

    Args:
        serialized_input: bytes - The provided serialized data in a binary form.
        serialization_format: str - The format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the format is not recognized or deemed unsafe.
        UnicodeDecodeError / json.JSONDecodeError / configparser.Error: If decoding/parsing fails.
    """
    # Validate argument types with logging for auditing
    if not isinstance(serialized_input, bytes) or not isinstance(serialization_format, str):
        logger.error(
            "Deserialization input validation failed: expected (bytes, str), got (%s, %s)",
            type(serialized_input).__name__,
            type(serialization_format).__name__,
        )
        if not isinstance(serialized_input, bytes):
            raise TypeError("serialized_input must be of type 'bytes'")
        raise TypeError("serialization_format must be of type 'str'")

    fmt = serialization_format.strip().lower()

    # Guard clause: reject unsupported/unsafe formats immediately
    if fmt not in approved_formats:
        logger.warning(
            "Deserialization rejected: unsupported/unsafe format=%r. Allowed formats: %s",
            serialization_format,
            ", ".join(approved_formats),
        )
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format!r}. "
            f"Allowed formats: {', '.join(approved_formats)}"
        )

    def _load_json(data: bytes):
        text = data.decode("utf-8")
        return json.loads(text)

    def _load_ini(data: bytes):
        text = data.decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(text)

        result = {}
        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())

        for section in parser.sections():
            result[section] = dict(parser.items(section))

        return result

    def _summarize_result(fmt_name: str, obj: Any) -> str:
        if fmt_name == "json":
            if isinstance(obj, dict):
                return f"type=dict, keys={len(obj)}"
            if isinstance(obj, list):
                return f"type=list, length={len(obj)}"
            return f"type={type(obj).__name__}"
        if fmt_name == "ini":
            sections = [k for k in obj.keys() if k != "DEFAULT"] if isinstance(obj, dict) else []
            has_default = isinstance(obj, dict) and "DEFAULT" in obj
            return f"sections={len(sections)}, default={has_default}"
        return f"type={type(obj).__name__}"

    handlers = {
        "json": _load_json,
        "ini": _load_ini,
    }

    handler = handlers.get(fmt)
    if handler is None:
        # Defensive guard; fmt should always be in handlers if approved.
        logger.error(
            "Deserialization failed due to missing handler for approved format=%s", fmt
        )
        raise ValueError(
            f"Unsupported or unsafe serialization format encountered: {serialization_format!r}"
        )

    logger.info(
        "Deserialization attempt: format=%s, size=%d bytes",
        fmt,
        len(serialized_input),
    )

    try:
        result = handler(serialized_input)
        summary = _summarize_result(fmt, result)
        logger.info("Deserialization succeeded: format=%s; %s", fmt, summary)
        return result
    except Exception as exc:
        logger.exception("Deserialization failed: format=%s; error=%s", fmt, str(exc))
        raise
