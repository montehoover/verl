"""
HTML parsing utilities for extracting tags using regular expressions.
"""

import re


# Pre-compiled regex to match start tags, end tags, and self-closing tags.
# Excludes comments (<!-- -->), doctypes (<!DOCTYPE ...>), and processing
# instructions (<? ... ?>).
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?>")


def get_html_tags(html_input: str) -> list:
    """
    Extract HTML tags from a provided string using a regular expression.

    Args:
        html_input: An HTML-formatted string.

    Returns:
        A list of matched HTML tags (as strings) found in the input.
    """
    try:
        html_text = html_input if isinstance(html_input, str) else str(html_input)

        if not html_text:
            return []

        matched_tags = HTML_TAG_PATTERN.findall(html_text)
        return matched_tags

    except Exception:
        # Do not raise exceptions under normal operation; return empty list on unexpected errors.
        return []
