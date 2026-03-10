from html import escape

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def sanitize_content(content: str) -> str:
    """
    Return a sanitized version of the content string safe for HTML display.
    Replaces &, <, and > with their HTML-safe equivalents.
    """
    return escape(content, quote=False)

def create_article_template(title: str, content: str) -> str:
    """
    Return a simple HTML string for an article with the given title and content.

    The format is: <h2>{title}</h2><p>{content}</p>
    """
    return f"<h2>{title}</h2><p>{content}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    """
    Return a formatted HTML string for a blog post using BLOG_FORMAT.
    Sanitizes the subject and description before formatting.
    """
    return BLOG_FORMAT.format(
        subject=sanitize_content(subject),
        description=sanitize_content(description),
    )
