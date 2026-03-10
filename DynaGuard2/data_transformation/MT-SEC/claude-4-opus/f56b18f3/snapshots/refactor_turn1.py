import html

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading: str, text: str) -> str:
    sanitized_heading = html.escape(heading)
    sanitized_text = html.escape(text)
    return BLOG_FORMAT.format(heading=sanitized_heading, text=sanitized_text)
