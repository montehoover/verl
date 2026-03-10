import re
from typing import Optional

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
