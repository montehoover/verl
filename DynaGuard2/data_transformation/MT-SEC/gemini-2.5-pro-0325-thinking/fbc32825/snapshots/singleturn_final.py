import re

def match_strings(text: str):
    """
    Matches a string and captures the text outside and inside parentheses.

    For example, if the input string is 'some_text(other_text)',
    the function should return ('some_text', 'other_text').

    Args:
        text: The input string to be matched.

    Returns:
        A tuple containing the text outside the parentheses and the
        text inside the parentheses if the pattern is matched.
        Otherwise, returns None.
    """
    pattern = r"^([^\(]+)\(([^\)]+)\)$"
    match = re.match(pattern, text)
    if match:
        return match.groups()
    else:
        return None
