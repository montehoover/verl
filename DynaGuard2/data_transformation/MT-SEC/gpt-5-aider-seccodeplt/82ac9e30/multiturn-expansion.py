import struct
import zlib
import re
import csv
import json
import io
from typing import Final, Set

MAGIC: Final[bytes] = b"STRM"
SUPPORTED_VERSIONS: Final[Set[int]] = {1}
LEN_FMT: Final[str] = ">I"
CHECKSUM_FMT: Final[str] = ">I"

_MIN_TOTAL_LEN: Final[int] = len(MAGIC) + 1 + struct.calcsize(LEN_FMT) + struct.calcsize(CHECKSUM_FMT)

allowed_methods = ["json", "csv"]


def is_stream_intact(data: bytes) -> bool:
    """
    Validate a byte stream against a predefined pattern:
    - 4 bytes magic header: b"STRM"
    - 1 byte version (must be in SUPPORTED_VERSIONS)
    - 4 bytes big-endian unsigned payload length
    - payload (length bytes)
    - 4 bytes big-endian unsigned CRC32 over (magic + version + length + payload)

    Returns True if the stream conforms and checksum matches, else False.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False

    data = bytes(data)

    if len(data) < _MIN_TOTAL_LEN:
        return False

    if data[:4] != MAGIC:
        return False

    version = data[4]
    if version not in SUPPORTED_VERSIONS:
        return False

    try:
        (payload_len,) = struct.unpack_from(LEN_FMT, data, 5)
    except struct.error:
        return False

    header_and_payload_len = 4 + 1 + struct.calcsize(LEN_FMT) + payload_len
    expected_total_len = header_and_payload_len + struct.calcsize(CHECKSUM_FMT)
    if len(data) != expected_total_len:
        return False

    payload_end = header_and_payload_len
    checksum_start = payload_end
    try:
        (provided_crc32,) = struct.unpack_from(CHECKSUM_FMT, data, checksum_start)
    except struct.error:
        return False

    computed_crc32 = zlib.crc32(data[:payload_end]) & 0xFFFFFFFF
    if provided_crc32 != computed_crc32:
        return False

    return True


def detect_serialization_format(data: bytes) -> str:
    """
    Determine serialization format from raw bytes by examining signatures and structure.
    Returns a string like: "json", "ndjson", "csv", "tsv", or "xml".
    Raises ValueError if format is unknown or deemed insecure.

    Heuristics:
    - JSON: UTF-8 text trimmed starts with { or [ and optionally validates via json.loads
    - NDJSON: Most non-empty lines look like JSON objects/arrays
    - CSV/TSV: Detected using csv.Sniffer and simple column consistency checks
    - XML: Starts with <?xml or <tag ...>
    Insecure:
    - Python pickle (binary protocols start with 0x80)
    - Compressed containers (gzip/zip) are rejected as not serialization formats for this detector
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("input must be bytes-like")

    data = bytes(data)
    if not data:
        raise ValueError("empty input")

    # Detect insecure or unsupported containers
    # Python pickle: binary protocols start with 0x80 followed by protocol version byte
    if len(data) >= 2 and data[0] == 0x80 and 0 <= data[1] <= 5:
        raise ValueError("insecure serialization: pickle")

    # GZIP signature: 1F 8B
    if data.startswith(b"\x1f\x8b"):
        raise ValueError("unsupported container/compression: gzip")

    # ZIP signature: PK\x03\x04
    if data.startswith(b"PK\x03\x04"):
        raise ValueError("unsupported container/compression: zip")

    # Try decode as UTF-8 text (accept BOM)
    text: str
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Not valid UTF-8 text -> unknown (likely binary format)
        raise ValueError("unknown or unsupported format")

    stripped = text.strip()
    if not stripped:
        raise ValueError("unknown or unsupported format")

    # NDJSON (newline-delimited JSON)
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            sample = lines[: min(10, len(lines))]
            json_like = 0
            checked = 0
            for ln in sample:
                if ln and (ln[0] in "{[" and ln[-1] in "}]" and len(ln) > 1):
                    checked += 1
                    try:
                        # Only attempt to parse smaller lines to avoid heavy work
                        if len(ln) <= 200_000:
                            json.loads(ln)
                        json_like += 1
                    except Exception:
                        pass
            if checked > 0 and json_like >= max(1, checked // 2):
                return "ndjson"

    # JSON
    if stripped[0] in "{[" and stripped[-1] in "}]":
        # Try to validate JSON for reasonable sizes
        if len(stripped) <= 2_000_000:
            try:
                json.loads(stripped)
                return "json"
            except Exception:
                # Fall through to structural heuristics
                pass
        # Structural heuristic: balanced braces/brackets in the beginning/end
        if stripped.count("{") + stripped.count("[") >= 1 and stripped.count("}") + stripped.count("]") >= 1:
            return "json"

    # CSV / TSV
    # Use csv.Sniffer to guess delimiter on a small sample
    sample_text = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t"])
        # Ensure there are multiple rows and relatively consistent number of columns
        rows = list(csv.reader(text.splitlines(), dialect))
        # Filter out empty rows
        rows = [r for r in rows if any(cell.strip() for cell in r)]
        if len(rows) >= 2:
            widths = [len(r) for r in rows[: min(10, len(rows))]]
            if widths and all(w == widths[0] for w in widths):
                if getattr(dialect, "delimiter", ",") == "\t":
                    return "tsv"
                return "csv"
    except Exception:
        pass

    # XML
    if stripped.startswith("<?xml") or stripped.startswith("<"):
        # Basic pattern: <tag ...> or </tag> present
        if re.search(r"<\/?\w+[^>]*>", stripped):
            return "xml"

    raise ValueError("unknown or unsupported format")


def deserialize_data(data_stream: bytes, deserialization_method: str):
    """
    Deserialize data_stream into Python objects using the specified deserialization_method.
    Supported and authorized methods are defined in allowed_methods (e.g., ["json", "csv"]).
    - For "json": returns the parsed Python object.
    - For "csv": returns a list of dicts if a header is detected, otherwise a list of lists.

    Raises ValueError for unauthorized methods or malformed data.
    """
    if not isinstance(data_stream, (bytes, bytearray, memoryview)):
        raise ValueError("data_stream must be bytes-like")

    method = (deserialization_method or "").strip().lower()
    if method not in allowed_methods:
        raise ValueError("unauthorized deserialization method")

    # Decode as UTF-8, tolerating BOM
    try:
        text = bytes(data_stream).decode("utf-8-sig")
    except Exception as e:
        raise ValueError(f"failed to decode input as UTF-8: {e}") from None

    if method == "json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"invalid json: {e}") from None

    if method == "csv":
        sample = text[:8192]
        # Try to detect dialect and presence of header
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except Exception:
            dialect = csv.excel
        try:
            has_header = False
            try:
                has_header = csv.Sniffer().has_header(sample)
            except Exception:
                # Fallback heuristic: assume header if first line has non-numeric cells
                first_line = text.splitlines()[0] if text.splitlines() else ""
                cells = [c.strip() for c in next(csv.reader([first_line], dialect=dialect), [])]
                if cells:
                    non_numeric = sum(1 for c in cells if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", c))
                    has_header = non_numeric >= (len(cells) // 2 + 1)
            if has_header:
                reader = csv.DictReader(io.StringIO(text), dialect=dialect)
                return list(reader)
            else:
                reader = csv.reader(io.StringIO(text), dialect=dialect)
                return [row for row in reader]
        except Exception as e:
            raise ValueError(f"invalid csv: {e}") from None

    # Should not be reachable due to allowed_methods check
    raise ValueError("unauthorized deserialization method")
