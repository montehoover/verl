def is_valid_string(value: str) -> bool:
    """
    Check if the input is a non-empty string that contains no space characters.
    """
    if not isinstance(value, str):
        return False
    return bool(value) and ' ' not in value
