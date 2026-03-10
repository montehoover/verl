import re

# Pre-compile the pattern to match: <outside>(<inside>) with the entire string constrained.
# Non-greedy for outside and inside to capture minimal text around a single pair of parentheses.
_PAREN_PATTERN = re.compile(r'^(.*?)\((.*?)\)$', re.DOTALL)


def match_strings(text: str):
    """
    Match a string and capture the text outside the parentheses and the text inside the parentheses.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: If the pattern matches, returns a tuple (outside, inside).
        Otherwise, returns None.

    Notes:
        - This function does not raise exceptions; invalid inputs or non-matching patterns return None.
    """
    if not isinstance(text, str):
        return None

    match = _PAREN_PATTERN.match(text)
    if not match:
        return None

    return match.group(1), match.group(2)
