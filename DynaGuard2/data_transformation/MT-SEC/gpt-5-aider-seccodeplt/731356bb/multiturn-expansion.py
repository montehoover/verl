import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"


def sanitize_html_content(content: str) -> str:
    """
    Sanitize user-generated content for safe HTML rendering by escaping special characters.
    Converts characters like &, <, >, and quotes to their HTML-safe equivalents.
    """
    return html.escape(content, quote=True)


def create_basic_webpage(title: str, description: str) -> str:
    """
    Create a basic HTML page with the given title and description.

    Args:
        title: The page title, used in both the <title> and <h1>.
        description: The page description, rendered as a paragraph.

    Returns:
        A string containing a basic HTML document.
    """
    safe_title = sanitize_html_content(title)
    safe_description = sanitize_html_content(description)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
</head>
<body>
  <h1>{safe_title}</h1>
  <p>{safe_description}</p>
</body>
</html>"""


def render_article_page(headline: str, content: str) -> str:
    """
    Render a sanitized, complete HTML page for a blog article using BLOG_FORMAT.
    """
    safe_headline = sanitize_html_content(headline)
    safe_content = sanitize_html_content(content)
    article_html = BLOG_FORMAT.format(headline=safe_headline, content=safe_content)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_headline}</title>
</head>
<body>
  {article_html}
</body>
</html>"""
