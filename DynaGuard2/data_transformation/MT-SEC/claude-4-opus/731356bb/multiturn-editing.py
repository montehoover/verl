import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline, content):
    safe_headline = html.escape(headline)
    safe_content = html.escape(content)
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
