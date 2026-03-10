import re
from typing import Optional, Tuple

def extract_inside_parentheses(input_text: str) -> Optional[str]:
    """
    Extract the first occurrence of text inside parentheses from the given string.

    Args:
        input_text (str): The string to search.

    Returns:
        Optional[str]: The text inside the first pair of parentheses, or None if not found.
    """
    match = re.search(r'\(([^()]*)\)', input_text)
    return match.group(1) if match else None


def capture_text_parts(input_text: str) -> Optional[Tuple[str, str]]:
    """
    Capture the text outside and inside the first pair of parentheses.

    Args:
        input_text (str): The string to search.

    Returns:
        Optional[Tuple[str, str]]: A tuple (outside_text, inside_text) where outside_text
        is the concatenation of the text before and after the first pair of parentheses,
        and inside_text is the text within those parentheses. Returns None if no match.
    """
    pattern = re.compile(r'^(.*?)[(]([^()]*)[)](.*)$', re.DOTALL)
    match = pattern.search(input_text)
    if not match:
        return None
    before, inside, after = match.groups()
    outside = f"{before}{after}"
    return outside, inside
