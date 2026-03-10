def add_header(header_value: str) -> str:
    if all(ch.isalnum() or ch == ' ' for ch in header_value):
        return f"Custom-Header: {header_value}"
    return "Invalid header value"
