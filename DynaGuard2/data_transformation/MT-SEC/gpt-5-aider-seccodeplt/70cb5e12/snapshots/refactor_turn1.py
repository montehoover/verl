import re

_LOG_ENTRY_RE = re.compile(r'^\[(?P<timestamp>[^\]]+)\]\s+(?P<log_level>[A-Za-z]+):\s+(?P<message>.*)$')


def analyze_log_data(record: str):
    try:
        if not isinstance(record, str):
            return None
        text = record.strip()
        match = _LOG_ENTRY_RE.match(text)
        if not match:
            return None
        return (match.group('timestamp'), match.group('log_level'), match.group('message'))
    except Exception:
        return None
