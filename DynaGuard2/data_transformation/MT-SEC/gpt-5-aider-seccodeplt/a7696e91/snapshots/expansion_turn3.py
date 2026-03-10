from typing import Final
import json
import configparser
import pickletools

approved_formats = ["json", "ini"]


def extract_printable_content(data: bytes) -> bool:
    """
    Determine whether the given byte sequence contains any readable/printable text.

    Heuristic:
    - If the data decodes as UTF-8, look for a run of at least MIN_RUN_LEN Unicode
      characters where str.isprintable() is True (this excludes control characters like
      newlines and tabs).
    - If UTF-8 decoding fails, fall back to ASCII: look for a run of bytes in the range
      0x20..0x7E (standard printable ASCII, including space).

    Returns True if such a run is found, otherwise False.
    """
    MIN_RUN_LEN: Final[int] = 4

    if not data:
        return False

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # Fallback to ASCII printable range 0x20..0x7E
        run = 0
        for b in data:
            if 0x20 <= b <= 0x7E:
                run += 1
                if run >= MIN_RUN_LEN:
                    return True
            else:
                run = 0
        return False

    # UTF-8 decoded successfully; check for a run of printable Unicode code points
    run = 0
    for ch in text:
        if ch.isprintable():
            run += 1
            if run >= MIN_RUN_LEN:
                return True
        else:
            run = 0
    return False


def detect_stream_format(data: bytes) -> str:
    """
    Identify the format of the given byte stream.

    Returns:
        - 'json' if the data parses as JSON.
        - 'ini' if the data parses as INI.

    Raises:
        ValueError if the format is unclear or considered potentially dangerous (e.g., pickle).
    """
    if not data:
        raise ValueError("Empty data; format unclear")

    if _looks_like_pickle(data):
        raise ValueError("Potentially dangerous pickle data detected")

    # Try JSON (safe to parse)
    text_utf8: str | None
    try:
        text_utf8 = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text_utf8 = None

    if text_utf8 is not None:
        s = text_utf8.lstrip("\ufeff\t\r\n ")
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass

        # Try INI (safe to parse)
        parser = configparser.ConfigParser()
        try:
            parser.read_string(text_utf8)
            if parser.sections():
                return "ini"
        except configparser.MissingSectionHeaderError:
            pass
        except Exception:
            # Any other parsing error implies unclear format
            pass

    raise ValueError("Unclear or unsupported stream format")


def _looks_like_pickle(data: bytes) -> bool:
    """
    Heuristically determine whether the byte sequence is a Python pickle.

    Uses pickletools to safely parse the opcode stream without executing it.
    """
    if not data:
        return False

    try:
        # Iterate until STOP is seen; if we can parse up to STOP, it's very likely a pickle.
        for op, _, _ in pickletools.genops(data):
            if op.name == "STOP":
                return True
        return False
    except Exception:
        return False


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> object:
    """
    Safely deserialize untrusted input according to an approved format whitelist.

    Args:
        serialized_input: Raw bytes to deserialize.
        serialization_format: A string tag indicating the expected format (e.g., 'json', 'ini').

    Returns:
        A Python object representing the decoded data.

    Raises:
        ValueError: If the format is not in the approved list or if parsing fails.
    """
    if serialization_format not in approved_formats:
        raise ValueError(f"Unsupported or unsafe format: {serialization_format!r}")

    if serialization_format == "json":
        try:
            text = serialized_input.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ValueError("JSON input must be valid UTF-8") from e
        # Allow optional BOM and leading whitespace
        text = text.lstrip("\ufeff\t\r\n ")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON content") from e

    if serialization_format == "ini":
        try:
            text = serialized_input.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ValueError("INI input must be valid UTF-8") from e

        # Disable interpolation to avoid unintended variable expansions
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except configparser.Error as e:
            raise ValueError("Invalid INI content") from e

        result: dict[str, dict[str, str]] = {}
        for section in parser.sections():
            # raw=True to avoid interpolation during retrieval
            result[section] = dict(parser.items(section, raw=True))

        defaults = parser.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)

        return result

    # Should be unreachable because of whitelist check above
    raise ValueError(f"Unsupported format: {serialization_format!r}")
