import html

BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def html_format_message(title: str, content: str) -> str:
    return f"<h1>{title}</h1><p>{content}</p>"

def display_blog_entry(headline: str, text: str) -> str:
    safe_headline = html.escape(headline, quote=True)
    safe_text = html.escape(text, quote=True)
    return BLOG_FORMAT.format(headline=safe_headline, text=safe_text)
