import re

def parse_html_tags(html: str):
    """
    Extract all HTML tags from the given string using a regular expression.

    The function returns the inner content of each tag (without angle brackets),
    including attributes and leading '/' in closing tags. It ignores comments,
    DOCTYPE/CDATA, and processing instructions. It does not raise exceptions.

    Args:
        html (str): The input HTML string to be parsed.

    Returns:
        list[str]: A list of strings, each representing an HTML tag found.
    """
    try:
        if not isinstance(html, str) or not html:
            return []

        # Capture everything between '<' and '>' non-greedily, trimming outer spaces.
        candidates = re.findall(r'<\s*([^>]+?)\s*>', html)

        tags = []
        for candidate in candidates:
            s = candidate.strip()
            if not s:
                continue

            # Skip comments and directives (e.g., <!-- ... -->, <!DOCTYPE ...>, <?xml ...?>, <![CDATA[ ... ]]>)
            ls = s.lstrip()
            if ls.startswith('!--') or ls.startswith('!') or ls.startswith('?') or ls.startswith('--'):
                continue
            if s.endswith('--'):
                continue

            tags.append(s)

        return tags
    except Exception:
        # Do not raise any exceptions; return an empty list on error.
        return []
