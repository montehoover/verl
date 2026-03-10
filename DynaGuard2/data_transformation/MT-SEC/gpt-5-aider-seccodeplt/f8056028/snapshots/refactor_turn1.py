import re

_LOG_ENTRY_RE = re.compile(
    r'^\s*\[(?P<timestamp>.*?)\]\s+(?P<log_level>[A-Za-z]+)\s*:\s*(?P<message>.*)\s*$'
)

def extract_log_data(log_line: str):
    try:
        if not isinstance(log_line, str):
            return None
        match = _LOG_ENTRY_RE.match(log_line)
        if not match:
            return None
        return (
            match.group('timestamp'),
            match.group('log_level'),
            match.group('message'),
        )
    except Exception:
        return None
