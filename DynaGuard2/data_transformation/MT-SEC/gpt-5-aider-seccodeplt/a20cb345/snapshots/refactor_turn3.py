"""Utilities for safely rendering user-submitted blog content to HTML.

This module provides a simple function to sanitize and format a blog post
using a predefined HTML template. It ensures that any user-provided text is
HTML-escaped before insertion to mitigate XSS and HTML injection risks.

Logging:
    The module logs key steps in the publishing process:
    - The subject of the blog article being processed.
    - A confirmation when the article has been formatted and is ready for display.
"""

import logging
from html import escape


# Module-level logger; configuration is expected to be handled by the application.
logger = logging.getLogger(__name__)

# Predefined string template for rendering a blog post. The placeholders are
# populated with sanitized values for the subject (title) and description (body).
BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"


def publish_blog_article(subject: str, description: str) -> str:
    """Sanitize and format a user-submitted blog post for safe display.

    This function accepts a blog post title and body, ensures both are strings,
    escapes any HTML-sensitive characters to prevent injection (e.g., XSS),
    and renders the result using the global BLOG_FORMAT template. It logs the
    subject being processed and confirms when the output is ready.

    Args:
        subject: The heading/title of the blog post.
        description: The main text content/body of the blog post.

    Returns:
        A string containing the sanitized and formatted blog post HTML.
    """
    # Normalize inputs to strings to avoid type errors during processing.
    if not isinstance(subject, str):
        subject = "" if subject is None else str(subject)

    if not isinstance(description, str):
        description = "" if description is None else str(description)

    # Log the start of the publishing process with the provided subject.
    logger.info("Publishing blog article: subject=%r", subject)

    # Escape HTML special characters to prevent injection/XSS vulnerabilities.
    # The `quote=True` argument also escapes quotation marks.
    safe_subject = escape(subject, quote=True)
    safe_description = escape(description, quote=True)

    # Render the final HTML using the predefined, trusted template.
    result = BLOG_FORMAT.format(
        subject=safe_subject,
        description=safe_description,
    )

    # Log a confirmation that the article is formatted and ready for display.
    logger.info(
        "Blog article formatted and ready for display: subject=%r",
        subject,
    )

    return result
