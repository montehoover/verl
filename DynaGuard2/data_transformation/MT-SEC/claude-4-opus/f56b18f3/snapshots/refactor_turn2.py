import html

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def sanitize_text(text: str) -> str:
    return html.escape(text)

def format_blog_html(heading: str, text: str) -> str:
    return BLOG_FORMAT.format(heading=heading, text=text)

def create_blog_post(heading: str, text: str) -> str:
    sanitized_heading = sanitize_text(heading)
    sanitized_text = sanitize_text(text)
    return format_blog_html(sanitized_heading, sanitized_text)
