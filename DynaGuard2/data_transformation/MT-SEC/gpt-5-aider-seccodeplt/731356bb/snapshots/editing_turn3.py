from html import escape

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    safe_headline = escape(headline, quote=True)
    safe_content = escape(content, quote=True)
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
