import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def format_blog_entry(headline: str, content: str) -> str:
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
