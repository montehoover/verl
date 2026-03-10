"""
Utilities for constructing an XSS-safe HTML representation of a user's forum post.

This module exposes:
- render_tags_html: Pure function that renders a list of tags into HTML-safe <span> elements.
- create_post_html: Pure function (w.r.t. its return value) that assembles a full post using a template,
  with appropriate escaping to prevent XSS. It also logs when a post HTML is created to aid debugging.
"""

import logging
from html import escape
from typing import List

# Configure a module-level logger. The application can configure handlers/levels as needed.
logger = logging.getLogger(__name__)

# Pre-existing HTML template string for formatting a user's forum post.
# Placeholders:
# - {post_title}: Escaped post title
# - {author_name}: Escaped author name
# - {content}: Escaped main content
# - {tags_html}: Concatenated <span> elements for tags
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"


def render_tags_html(post_tags: List[str]) -> str:
    """
    Convert a list of tags into a concatenated string of HTML <span> elements, XSS-safe.

    Each tag is:
      1) Converted to string (to be robust if non-strings sneak in),
      2) Escaped with html.escape to neutralize any HTML/XML special characters,
      3) Wrapped in a <span> tag without attributes.

    Args:
        post_tags: A list of tags related to the post.

    Returns:
        A single string containing each tag wrapped in <span>...</span> with no separators.

    Examples:
        >>> render_tags_html(["python", "security"])
        '<span>python</span><span>security</span>'
    """
    # Be defensive: allow graceful handling of falsy inputs (e.g., None or empty list).
    tags = post_tags or []

    # Escape each tag to prevent XSS and wrap in <span>.
    safe_tags = [escape(str(tag), quote=True) for tag in tags]
    tags_html = "".join(f"<span>{tag}</span>" for tag in safe_tags)

    # Debug-level log with structural details (avoid logging raw unescaped content).
    logger.debug("Rendered tags HTML (count=%d, length=%d)", len(safe_tags), len(tags_html))
    return tags_html


def create_post_html(author_name: str, post_title: str, content: str, post_tags: List[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    All user-supplied fields are escaped using html.escape with quote=True to prevent
    injection of HTML/JS content. Tags are rendered via render_tags_html.

    Args:
        author_name: The name of the post's author.
        post_title: The headline of the user's post.
        content: The primary text content of the user's post.
        post_tags: A list of tags related to the post.

    Returns:
        A string containing the sanitized HTML representation of the post.

    Notes:
        - This function logs at INFO level when a post HTML is created, including
          the title/author (escaped), tag count, and total HTML length to aid debugging.
        - Avoid logging raw content to minimize risk of leaking sensitive data.
    """
    # Escape core fields. Use empty string when None to avoid "None" literal appearing.
    safe_author = escape(author_name if author_name is not None else "", quote=True)
    safe_title = escape(post_title if post_title is not None else "", quote=True)
    safe_content = escape(content if content is not None else "", quote=True)

    # Render the tags separately to keep responsibilities separated and testable.
    tag_count = len(post_tags) if post_tags else 0
    tags_html = render_tags_html(post_tags)

    # Fill the predefined HTML template with safe, escaped values.
    html_str = USER_POST_FORMAT.format(
        post_title=safe_title,
        author_name=safe_author,
        content=safe_content,
        tags_html=tags_html,
    )

    # Record a concise creation log for observability and debugging purposes.
    logger.info(
        "Post HTML created (title=%r, author=%r, tags=%d, length=%d)",
        safe_title,
        safe_author,
        tag_count,
        len(html_str),
    )

    return html_str
