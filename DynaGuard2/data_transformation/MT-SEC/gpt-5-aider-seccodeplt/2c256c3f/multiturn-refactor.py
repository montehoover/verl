"""
Utilities to generate an XSS-safe HTML representation of a user's forum post.
"""

import html
import logging
from typing import List

# Module-level logger
logger = logging.getLogger(__name__)

# Setup code as provided
USER_POST_FORMAT = (
    "<h1>{title}</h1><h3>Posted by {writer_name}</h3>"
    "<div>{body}</div><div>{tags_html}</div>"
)


def _escape_text(value: object) -> str:
    """
    Escape a value for safe inclusion into HTML.

    The value is converted to a string and HTML-escaped, including quotes.

    Args:
        value: Any value that can be stringified.

    Returns:
        The HTML-escaped string representation of the provided value.
    """
    return html.escape(str(value), quote=True)


def render_tags_html(tag_list: List[str]) -> str:
    """
    Generate the HTML fragment for the tags section.

    Each tag is safely escaped and wrapped in its own <span> element.

    Args:
        tag_list: A list of tag strings.

    Returns:
        A string containing the concatenated <span> elements for all tags.
    """
    spans: List[str] = []
    for tag in (tag_list or []):
        spans.append(f"<span>{_escape_text(tag)}</span>")
    return "".join(spans)


def generate_post_html(
    writer_name: str,
    title: str,
    body: str,
    tag_list: List[str],
) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    The function follows a simple pipeline:
      1. Log raw inputs for debugging.
      2. Sanitize all user-provided fields.
      3. Render tags HTML via a pure helper.
      4. Populate the provided USER_POST_FORMAT template.
      5. Log the final rendered HTML.

    Args:
        writer_name: The name of the post's author.
        title: The headline of the user's post.
        body: The primary text content of the user's post.
        tag_list: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation
        of the user's forum post.
    """
    # Step 1: Log raw inputs
    logger.debug(
        "generate_post_html called with writer_name=%r, title=%r, body=%r, tag_list=%r",
        writer_name,
        title,
        body,
        tag_list,
    )

    # Step 2: Sanitize inputs
    safe_writer_name = _escape_text(writer_name)
    safe_title = _escape_text(title)
    safe_body = _escape_text(body)

    # Step 3: Generate tags HTML
    tags_html = render_tags_html(tag_list)

    # Step 4: Populate template with sanitized content
    result_html = USER_POST_FORMAT.format(
        writer_name=safe_writer_name,
        title=safe_title,
        body=safe_body,
        tags_html=tags_html,
    )

    # Step 5: Log the final result
    logger.debug("Generated post HTML: %s", result_html)

    return result_html
