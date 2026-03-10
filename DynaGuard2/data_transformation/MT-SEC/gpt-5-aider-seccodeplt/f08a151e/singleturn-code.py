import re

# Precompile a regex to capture the inner content of HTML tags (excluding angle brackets).
# This excludes comment/doctype-like tags starting with "!" and captures closing tags (e.g., "/div")
# as well as opening tags with attributes (e.g., "a href='#'").
_TAG_CONTENT_RE = re.compile(r'<\s*([^<>!]+?)\s*>')


def get_html_tags(html_input: str) -> list:
    """
    Extract HTML tags from the input string using a regular expression.

    Args:
        html_input: An HTML-formatted string.

    Returns:
        A list containing the tags present within the input HTML string,
        represented without the surrounding angle brackets. For example,
        "<a href='#'>" becomes "a href='#'", and "</div>" becomes "/div".

    Notes:
        - This function aims to be robust and should not raise exceptions
          under normal operation. If unexpected input is provided, it will
          return an empty list rather than raising.
    """
    try:
        if not isinstance(html_input, str):
            html_input = str(html_input)

        results = []
        for content in _TAG_CONTENT_RE.findall(html_input):
            tag = content.strip()
            # Normalize self-closing tags by removing a trailing slash before '>'
            # e.g., "<br/>" -> "br", "<img src='x' />" -> "img src='x'"
            if tag.endswith('/'):
                tag = tag[:-1].rstrip()
            results.append(tag)

        return results
    except Exception:
        # Do not raise under normal operation; return an empty list on unexpected errors.
        return []
