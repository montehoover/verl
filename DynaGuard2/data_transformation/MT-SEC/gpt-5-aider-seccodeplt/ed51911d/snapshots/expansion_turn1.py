import re
from typing import List


def parse_placeholders(html: str) -> List[str]:
    """
    Parse an HTML string for placeholders formatted as {{...}} and return a list
    of all placeholder names found. Supports nested placeholders.

    Example:
        "{{outer {{inner}} }}" -> ["inner", "outer {{inner}}"]

    Args:
        html: The HTML string to parse.

    Returns:
        A list of placeholder contents (names), in the order they are closed
        (inner placeholders appear before their containing outer placeholders).
    """
    results: List[str] = []
    stack: List[int] = []
    i = 0
    n = len(html)

    while i < n - 1:
        pair = html[i:i + 2]
        if pair == "{{":
            stack.append(i)
            i += 2
            continue
        if pair == "}}" and stack:
            start = stack.pop()
            content = html[start + 2:i]
            results.append(content.strip())
            i += 2
            continue
        i += 1

    return results
