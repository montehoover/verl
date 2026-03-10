import re

_HTTP_PROTOCOL_PATTERN = re.compile(r'^https?://', re.IGNORECASE)

def is_http_protocol(value: str) -> bool:
    return isinstance(value, str) and bool(_HTTP_PROTOCOL_PATTERN.match(value))
