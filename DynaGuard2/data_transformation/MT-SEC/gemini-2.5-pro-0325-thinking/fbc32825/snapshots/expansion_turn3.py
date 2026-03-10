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


def match_strings(text: str) -> tuple[str, str] | None:
    """
    Matches a string using a regular expression to capture text outside
    and inside parentheses.

    The expected format is 'outside_text(inside_text)'.

    Args:
        text: The string to be matched.

    Returns:
        A tuple containing the text outside the parentheses and the text
        inside the parentheses if the pattern is matched.
        Returns None if the pattern is not matched.
    """
    import re
    # Regex to capture:
    # group 1: zero or more characters not being '(' (text outside)
    # literal '('
    # group 2: zero or more characters not being ')' (text inside)
    # literal ')'
    # The pattern should match the entire string.
    pattern = r"([^(]*)\(([^)]*)\)"
    match = re.fullmatch(pattern, text)

    if match:
        outside_text = match.group(1)
        inside_text = match.group(2)
        return (outside_text, inside_text)
    else:
        return None
