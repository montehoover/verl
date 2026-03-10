from html import escape

BLOG_FORMAT: str = "<h2>{headline}</h2><div>{text}</div>"

def sanitize_html(text: str) -> str:
    """
    Return a version of the input string with special HTML characters escaped.
    """
    return escape(text, quote=True)

def create_html_paragraph(text: str) -> str:
    """
    Create an HTML string with the given text wrapped in a <p> element.
    The text is HTML-escaped to ensure safe rendering of user-generated content.
    """
    return f"<p>{sanitize_html(text)}</p>"

def display_blog_entry(headline: str, text: str) -> str:
    """
    Format and return a sanitized, HTML-formatted blog entry string.
    Expects a global BLOG_FORMAT with placeholders: {headline} and {text}.
    """
    safe_headline = sanitize_html(headline)
    body_html = create_html_paragraph(text)
    return BLOG_FORMAT.format(headline=safe_headline, text=body_html)
