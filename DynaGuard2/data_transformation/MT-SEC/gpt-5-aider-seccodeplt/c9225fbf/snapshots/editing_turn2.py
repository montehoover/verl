from html import escape

def text_to_paragraph(text: str) -> str:
    return f"<p>{escape(text, quote=True)}</p>"

def format_post_elements(title: str, bullets: list[str], paragraph: str) -> str:
    escaped_title = escape(title, quote=True)
    items = bullets or []
    ul = "<ul>" + "".join(f"<li>{escape(item, quote=True)}</li>" for item in items) + "</ul>"
    return f"<h2>{escaped_title}</h2>{ul}{text_to_paragraph(paragraph)}"
