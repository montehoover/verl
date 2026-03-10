import html

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"


def sanitize_content(text: str) -> str:
    """
    Sanitizes a string by replacing special HTML characters.

    Args:
        text: The input string to sanitize.

    Returns:
        The sanitized string with HTML special characters replaced.
    """
    return html.escape(text)


def create_article_template(title: str, content: str) -> str:
    """
    Creates a simple HTML string for an article.

    Args:
        title: The title of the article.
        content: The content of the article.

    Returns:
        An HTML string with the title in <h2> tags and content in <p> tags.
    """
    return f"<h2>{title}</h2><p>{content}</p>"


def publish_blog_article(subject: str, description: str) -> str:
    """
    Creates a formatted HTML string for a blog post.

    Args:
        subject: The subject/title of the blog post.
        description: The main content/body of the blog post.

    Returns:
        A formatted HTML string for the blog post, with content sanitized.
    """
    sanitized_subject = sanitize_content(subject)
    sanitized_description = sanitize_content(description)
    return BLOG_FORMAT.format(subject=sanitized_subject, description=sanitized_description)
