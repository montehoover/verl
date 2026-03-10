import re

def parse_html_tags(html: str) -> list[str]:
    """
    Extracts all HTML tags from a given string using a regular expression.

    Args:
        html: str, the input HTML string to be parsed.

    Returns:
        A list of strings, each representing an HTML tag found in the input.
        Returns an empty list if no tags are found or in case of an error.
    """
    try:
        # Regular expression to find HTML tags (e.g., <p>, <div>, <a href="...">, </p>)
        # This regex captures anything that starts with <, ends with >, and has one or more characters in between.
        # It's a simplified regex and might not cover all edge cases of HTML parsing.
        tags = re.findall(r"<[^>]+>", html)
        return tags
    except Exception:
        # As per requirement, do not raise exceptions. Return empty list on error.
        return []
