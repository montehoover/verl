def add_custom_header(name: str) -> str:
    """Returns a custom HTTP header string with the provided name."""
    return f"print({name})"
