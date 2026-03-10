import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def sanitize_text(text: str) -> str:
    return html.escape(text)

def format_blog_entry(headline: str, content: str) -> str:
    sanitized_headline = sanitize_text(headline)
    sanitized_content = sanitize_text(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
