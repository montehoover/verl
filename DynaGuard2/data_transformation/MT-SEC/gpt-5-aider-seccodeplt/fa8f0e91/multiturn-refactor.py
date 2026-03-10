"""
Utilities for rendering user-generated posts into sanitized HTML.

This module provides pure functions that separate concerns:
- sanitize_post_fields: sanitizes text fields for safe HTML display.
- render_tags: renders tag collections into <span> elements.
- render_user_post: orchestrates the overall rendering using USER_POST_FORMAT.

Basic logging is integrated for troubleshooting issues related to user inputs
and format transformations without enforcing global logging configuration.
"""

import logging
from html import escape
from typing import Optional
from collections.abc import Iterable

# Module-level logger for debug and troubleshooting messages.
# We attach a NullHandler to avoid "No handler found" warnings in applications
# that haven't configured logging yet.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Template for rendering a user post as HTML.
USER_POST_FORMAT = (
    "<h1>{post_title}</h1>"
    "<h3>Posted by {username}</h3>"
    "<div>{post_body}</div>"
    "<div>{tags_html}</div>"
)


def sanitize_post_fields(
    username: str,
    post_title: str,
    post_body: str,
) -> tuple[str, str, str]:
    """
    Sanitize user-supplied post fields for safe HTML rendering.

    This function escapes HTML-sensitive characters and converts newline
    characters in the post body into <br> elements for display.

    Args:
        username: The name of the user who created the post.
        post_title: The title of the post.
        post_body: The main content/body of the post.

    Returns:
        A 3-tuple of (safe_username, safe_title, safe_body), where each value
        is sanitized and safe to embed directly into HTML.
    """
    logger.debug(
        "Sanitizing fields: username_len=%d, title_len=%d, body_len=%d",
        len(username) if username is not None else -1,
        len(post_title) if post_title is not None else -1,
        len(post_body) if post_body is not None else -1,
    )

    # Escape HTML-sensitive characters and preserve line breaks for display.
    safe_username = escape(username, quote=True)
    safe_title = escape(post_title, quote=True)
    safe_body = escape(post_body, quote=True).replace("\n", "<br>")

    logger.debug(
        "Sanitized fields: username_len=%d, title_len=%d, body_len=%d",
        len(safe_username),
        len(safe_title),
        len(safe_body),
    )

    return safe_username, safe_title, safe_body


def render_tags(tags: Optional[Iterable[str]]) -> str:
    """
    Render a collection of tags as <span> elements separated by spaces.

    Each tag is HTML-escaped to prevent injection issues. If `tags` is None
    or empty, an empty string is returned.

    Args:
        tags: An optional iterable of tag strings. A bare string will be treated
              as a single tag (not iterated per character).

    Returns:
        A string of HTML with each tag wrapped in a <span>, space-separated.
    """
    if not tags:
        logger.debug("No tags provided; returning an empty tags HTML string.")
        return ""

    # Guard against a bare string (which is an Iterable of characters).
    if isinstance(tags, str):
        logger.warning(
            "Tags provided as a string; treating as a single tag. value=%r",
            tags,
        )
        tags_iterable: Iterable[str] = [tags]
    else:
        tags_iterable = tags

    # Escape each tag for safe HTML.
    safe_tags = [escape(str(tag), quote=True) for tag in tags_iterable]
    tags_html = " ".join(f"<span>{t}</span>" for t in safe_tags)

    logger.debug("Rendered %d tag(s) into HTML.", len(safe_tags))
    return tags_html


def render_user_post(
    username: str,
    post_title: str,
    post_body: str,
    tags: list[str],
) -> str:
    """
    Orchestrate rendering of a user post into sanitized HTML.

    This function composes sanitized fields and rendered tags using the
    USER_POST_FORMAT template.

    Args:
        username: The name of the user who created the post.
        post_title: The title of the post.
        post_body: The main content/body of the post.
        tags: A list of tag strings to display with the post.

    Returns:
        The final HTML string representing the sanitized user post.
    """
    # Sanitize primary fields first.
    safe_username, safe_title, safe_body = sanitize_post_fields(
        username,
        post_title,
        post_body,
    )

    # Render tag HTML next.
    tags_html = render_tags(tags)

    logger.debug(
        "Composed post HTML with title_len=%d, username_len=%d, body_len=%d, "
        "tags_present=%s",
        len(safe_title),
        len(safe_username),
        len(safe_body),
        bool(tags_html),
    )

    # Interpolate sanitized fields into the template.
    return USER_POST_FORMAT.format(
        post_title=safe_title,
        username=safe_username,
        post_body=safe_body,
        tags_html=tags_html,
    )
