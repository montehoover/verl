from html import escape


def format_blog_html(template: str, headline: str, content: str) -> str:
    """
    Pure function that formats a blog post into HTML using the provided template.

    Args:
        template: An HTML template string with {headline} and {content} placeholders.
        headline: Sanitized headline text to insert.
        content: Sanitized content text to insert.

    Returns:
        The HTML-formatted blog post string.
    """
    return template.format(headline=headline, content=content)


def render_article_page(headline: str, content: str) -> str:
    """
    Sanitize and format a blog post into HTML using the global BLOG_FORMAT template.

    Args:
        headline: the headline or title of the blog post.
        content: the full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post.
    """
    safe_headline = escape(headline, quote=True)
    safe_content = escape(content, quote=True)

    try:
        template = BLOG_FORMAT  # type: ignore[name-defined]
    except NameError:
        template = "<h1>{headline}</h1><p>{content}</p>"

    return format_blog_html(template, safe_headline, safe_content)
