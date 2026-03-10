import re

def grab_html_tags(html_code: str):
    """
    Scans the provided HTML string for tags using a regular expression and returns
    a list of all detected tags. The function is designed to avoid raising exceptions.
    """
    try:
        if not isinstance(html_code, str):
            html = "" if html_code is None else str(html_code)
        else:
            html = html_code

        pattern = re.compile(
            r'<!--.*?-->|'                  # HTML comments
            r'<!DOCTYPE[^>]*>|'             # DOCTYPE
            r'</?[A-Za-z][A-Za-z0-9:-]*'    # Opening or closing tag with name
            r'(?:\s+[^<>]*?)?'              # Optional attributes
            r'\s*/?>',                      # Optional whitespace and optional self-closing slash
            re.IGNORECASE | re.DOTALL
        )
        return pattern.findall(html) or []
    except Exception:
        return []
