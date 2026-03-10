"""
Module to handle blog post publishing.

This module provides functionality to take user-submitted blog post content,
sanitize it to prevent common web vulnerabilities, and format it into
an HTML string suitable for display on a webpage.
"""
import html

# Predefined HTML template for blog posts.
# This template structures the blog post with a main heading (subject)
# and a paragraph for the main content (description).
BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"


def publish_blog_article(subject: str, description: str) -> str:
    """
    Sanitizes and formats a user-submitted blog post for safe web display.

    The function takes a subject (title) and a description (content) for the
    blog post. Both inputs are first sanitized using `html.escape` to
    prevent Cross-Site Scripting (XSS) vulnerabilities by escaping
    special HTML characters. The sanitized inputs are then formatted into an
    HTML string using a predefined template (`BLOG_FORMAT`).

    Args:
        subject: The heading (title) of the blog post. This will be
                 placed within <h1> tags.
        description: The main text content (body) of the blog post. This
                     will be placed within <p> tags.

    Returns:
        A string containing the HTML-formatted and sanitized blog post,
        ready for safe display on a webpage.
    """
    # Sanitize the subject and description to prevent XSS attacks.
    # html.escape replaces special characters (e.g., <, >, &)
    # with their corresponding HTML entities (e.g., &lt;, &gt;, &amp;).
    sanitized_subject = html.escape(subject)
    sanitized_description = html.escape(description)

    # Format the sanitized content using the predefined blog template.
    # The .format() method substitutes the placeholders in BLOG_FORMAT
    # with the sanitized subject and description.
    return BLOG_FORMAT.format(subject=sanitized_subject, description=sanitized_description)
