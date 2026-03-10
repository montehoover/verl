import csv
import json
import configparser
import xml.etree.ElementTree as ET


def is_ascii_printable(byte_stream):
    """
    Check whether the given byte_stream contains only ASCII printable characters.

    ASCII printable range: 0x20 (space) to 0x7E (~), inclusive.
    Returns:
        bool: True if all bytes are in the printable ASCII range, False otherwise.
    """
    try:
        data = bytes(byte_stream)
    except Exception:
        return False

    return all(32 <= b <= 126 for b in data)


def detect_format(byte_stream):
    """
    Detect common serialization formats from a bytes-like object.
    Returns a lowercase string format name (e.g., 'json', 'csv', 'xml').
    Raises ValueError if unidentified or deemed unsafe.

    Formats detected:
      - json
      - ndjson (newline-delimited JSON)
      - xml
      - csv
      - tsv
      - ini
      - avro (Obj\\x01 header)
      - parquet (PAR1 header/footer)
      - bplist (bplist00 header)

    Unsafe detections (raise):
      - python pickle (heuristic detection)
    """
    try:
        data = bytes(byte_stream)
    except Exception as e:
        raise ValueError(f"Input is not bytes-like: {e}")

    if not data:
        raise ValueError("Empty input")

    # Known binary signatures first
    if data.startswith(b'bplist00'):
        return 'bplist'

    if data.startswith(b'Obj\x01'):
        return 'avro'

    if len(data) >= 8 and data[:4] == b'PAR1' and data[-4:] == b'PAR1':
        return 'parquet'

    # Unsafe: heuristic detection of Python pickle (binary protocols)
    # Protocol byte 0x80 followed by version 0..5 (historically) and STOP '.' at the end
    if len(data) >= 2 and data[0] == 0x80 and data[1] in range(0, 6) and data.endswith(b'.'):
        raise ValueError("Unsafe serialization format detected: pickle")

    # Attempt text-based detections
    text = None

    # Detect BOM to choose decoding, otherwise try UTF-8 strictly
    try:
        if data.startswith(b'\xff\xfe\x00\x00'):
            text = data.decode('utf-32le')
        elif data.startswith(b'\x00\x00\xfe\xff'):
            text = data.decode('utf-32be')
        elif data.startswith(b'\xff\xfe'):
            text = data.decode('utf-16le')
        elif data.startswith(b'\xfe\xff'):
            text = data.decode('utf-16be')
        else:
            # utf-8-sig handles optional UTF-8 BOM
            text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = None

    if text is not None:
        s = text
        stripped = s.lstrip()

        # JSON
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                json.loads(s)
                return 'json'
            except Exception:
                pass

        # NDJSON (newline-delimited JSON)
        # Heuristic: at least 2 non-empty lines and each parses as JSON
        lines = [ln for ln in s.splitlines() if ln.strip()]
        if len(lines) >= 2:
            sample_lines = lines[:50]
            all_json = True
            for ln in sample_lines:
                ln_stripped = ln.strip()
                if not (ln_stripped.startswith('{') or ln_stripped.startswith('[')):
                    all_json = False
                    break
                try:
                    json.loads(ln_stripped)
                except Exception:
                    all_json = False
                    break
            if all_json:
                return 'ndjson'

        # XML
        if stripped.startswith('<?xml') or stripped.startswith('<'):
            try:
                ET.fromstring(s)
                return 'xml'
            except Exception:
                pass

        # INI
        try:
            cp = configparser.ConfigParser()
            cp.read_string(s)
            if cp.sections():
                return 'ini'
        except Exception:
            pass

        # CSV / TSV detection using csv.Sniffer
        try:
            sample = s[:4096]
            # Only attempt if there are at least 1-2 lines and contains typical delimiters
            if '\n' in sample and any(d in sample for d in [',', '\t', ';', '|']):
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', '\t', ';', '|'])
                # Validate that we can parse and get rows with >1 fields
                reader = csv.reader(s.splitlines(), dialect)
                multi_field_rows = 0
                for i, row in enumerate(reader):
                    if len(row) > 1:
                        multi_field_rows += 1
                    if i >= 10:
                        break
                if multi_field_rows >= 1:
                    if dialect.delimiter == '\t':
                        return 'tsv'
                    elif dialect.delimiter == ',':
                        return 'csv'
                    else:
                        # Other delimiters treated as CSV-like
                        return 'csv'
        except Exception:
            pass

    # If we reached here, either unidentified or potentially unsafe/ambiguous
    raise ValueError("Format not identified or deemed unsafe")
