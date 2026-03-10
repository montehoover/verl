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


def identify_parts(text: str) -> dict[str, str]:
    """
    Identifies and separates parts of a string formatted as 'prefix(suffix)'.

    Args:
        text: The string to be processed, expected in 'prefix(suffix)' format.

    Returns:
        A dictionary with keys 'prefix' and 'suffix'.
        Returns {'prefix': '', 'suffix': ''} if the format is not matched.
    """
    # prefix: The section of the string before the parentheses
    # suffix: The section of the string inside the parentheses
    import re
    match = re.fullmatch(r"([^(]*)\(([^)]*)\)", text)
    if match:
        prefix = match.group(1)
        suffix = match.group(2)
        return {"prefix": prefix, "suffix": suffix}
    return {"prefix": "", "suffix": ""}
