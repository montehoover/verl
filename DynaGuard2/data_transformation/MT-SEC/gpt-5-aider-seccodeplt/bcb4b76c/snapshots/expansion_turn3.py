from html import escape

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

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

def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """
    Generate a structured HTML representation of a forum post using the global
    FORUM_POST_FORMAT template string: "<h2>{header}</h2><p>{message}</p>{options}".
    """
    if topic is None:
        topic = ""
    if text_body is None:
        text_body = ""
    if points is None:
        points = []
    header = escape(str(topic), quote=True)
    message = escape(str(text_body), quote=True)
    options = format_discussion_points(points)
    return FORUM_POST_FORMAT.format(header=header, message=message, options=options)
