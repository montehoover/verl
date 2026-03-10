from html import escape

def text_to_paragraph(text: str) -> str:
    """
    Convert a text string into an HTML paragraph by wrapping it in <p> tags.
    Special characters are escaped to ensure safe HTML output.
    """
    return f"<p>{escape(text, quote=True)}</p>"

def format_post_elements(title: str, bullets: list[str], paragraph: str) -> str:
    """
    Format a post's elements into HTML:
    - Title wrapped in <h2>
    - Bullet points wrapped in <ul>/<li>
    - Paragraph wrapped in <p>
    Special characters are escaped to ensure safe HTML output.
    """
    safe_title = escape(title, quote=True)
    list_items = "".join(f"<li>{escape(item, quote=True)}</li>" for item in bullets)
    ul_block = f"<ul>{list_items}</ul>"
    paragraph_block = text_to_paragraph(paragraph)
    return f"<h2>{safe_title}</h2>{ul_block}{paragraph_block}"
