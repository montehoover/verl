import html

BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def display_blog_entry(headline: str, text: str) -> str:
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
