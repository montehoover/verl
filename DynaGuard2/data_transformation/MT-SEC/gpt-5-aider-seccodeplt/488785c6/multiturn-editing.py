import re

def add_header(header_value: str) -> str:
    if all(ch.isalnum() or ch == ' ' for ch in header_value):
        return f"Custom-Header: {header_value}"
    return "Invalid header value"

def add_safe_header(custom_value: str) -> str:
    value = "" if custom_value is None else str(custom_value)
    try:
        value = re.sub(r"[\r\n]+", " ", value)
        value = re.sub(r"[^\t\x20-\x7E\x80-\xFF]", "", value)
        value = re.sub(r"[\t ]+", " ", value).strip()
    except re.error:
        filtered_chars = []
        for ch in value:
            o = ord(ch)
            if ch in ("\r", "\n"):
                continue
            if ch == "\t" or ch == " " or 0x20 <= o <= 0x7E or 0x80 <= o <= 0xFF:
                filtered_chars.append(ch)
        value = "".join(filtered_chars)
        value = " ".join(value.split())
    return f"Custom-Header: {value}"
