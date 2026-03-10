import re

def match_strings(text):
    """
    Match a string and capture the text outside the first pair of parentheses
    and the text inside them.

    Args:
        text (str): The input string.

    Returns:
        tuple[str, str] | None: A tuple (outside_text, inside_text) if matched, else None.
    """
    if not isinstance(text, str):
        return None

    match = re.search(r'^(.*?)[(]([^()]*)[)](.*)$', text, flags=re.DOTALL)
    if not match:
        return None

    before, inside, after = match.groups()
    outside = f"{before}{after}"
    return outside, inside
