import re

def identify_html_tags(content: str) -> list:
    """
    Scans the provided HTML string for tags using a regular expression and returns a list
    of the tag contents (without angle brackets). The function is resilient and will not
    raise exceptions; invalid or non-string inputs yield an empty list.

    Args:
        content: An HTML string to scan.

    Returns:
        List[str]: A list of tag descriptors (e.g., 'div', '/div', "a href='#'").
    """
    if not isinstance(content, str):
        return []
    try:
        # Find everything between < and > that is not another < or >
        # This captures opening tags, closing tags, and self-closing tags.
        raw_tags = re.findall(r'<\s*([^<>]+?)\s*>', content, flags=re.DOTALL)
        tags = []
        for t in raw_tags:
            s = t.strip()
            # Skip HTML comments like <!-- ... -->
            if s.startswith('!--') and s.endswith('--'):
                continue
            # Normalize self-closing tags by removing a trailing slash (e.g., <br/> -> 'br')
            if not s.startswith('/'):
                s = re.sub(r'\s*/\s*$', '', s)
            tags.append(s)
        return tags
    except Exception:
        return []
