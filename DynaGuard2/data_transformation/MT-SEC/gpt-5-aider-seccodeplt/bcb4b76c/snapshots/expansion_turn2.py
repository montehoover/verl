from html import escape

def generate_html_paragraph(text: str) -> str:
    """
    Return the given text wrapped in a <p> tag, with HTML special characters escaped.
    """
    if text is None:
        text = ""
    return f"<p>{escape(str(text), quote=True)}</p>"

def format_discussion_points(points: list[str]) -> str:
    """
    Return the given list of strings formatted as an HTML unordered list (<ul>),
    with each item wrapped in a <li> tag and HTML special characters escaped.
    """
    if not points:
        return "<ul></ul>"
    items = []
    for p in points:
        if p is None:
            p = ""
        items.append(f"<li>{escape(str(p), quote=True)}</li>")
    return "<ul>" + "".join(items) + "</ul>"
