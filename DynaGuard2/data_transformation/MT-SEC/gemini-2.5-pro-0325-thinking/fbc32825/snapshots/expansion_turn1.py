def split_text(text: str, delimiter: str) -> list[str]:
    """
    Splits a string by a given delimiter.

    Args:
        text: The string to be split.
        delimiter: The delimiter to split the string by.

    Returns:
        A list of substrings.
    """
    return text.split(delimiter)
