if 'BLOG_FORMAT' not in globals():
    BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading: str, text: str) -> str:
    from html import escape
    safe_heading = escape(heading, quote=True)
    safe_text = escape(text, quote=True)
    return BLOG_FORMAT.format(heading=safe_heading, text=safe_text)
