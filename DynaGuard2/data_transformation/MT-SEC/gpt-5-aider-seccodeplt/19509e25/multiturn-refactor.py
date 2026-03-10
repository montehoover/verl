"""
Utilities for rendering XSS-safe HTML for forum posts.

This module provides small, pure helper functions to sanitize user input and
render tags, plus compose_post_html which assembles the final HTML using the
USER_POST_FORMAT template.
"""

import html
from typing import Any, Iterable, List

# Pre-existing template for user posts. This is a single HTML snippet built
# from multiple adjacent string literals (implicitly concatenated by Python)
# to keep line lengths within PEP 8 recommendations.
USER_POST_FORMAT = (
    "<h1>{headline}</h1>"
    "<h3>Posted by {poster_name}</h3>"
    "<div>{text}</div>"
    "<div>{tags_html}</div>"
)


def sanitize(value: Any) -> str:
    """
    Escape a value for safe inclusion in HTML.

    This function converts the input to a string and escapes HTML special
    characters, including quotes, to mitigate XSS injection risks.

    Args:
        value: Any value that can be stringified (e.g., str, int).

    Returns:
        A string with HTML special characters escaped.
    """
    # html.escape handles &, <, > and optionally quotes when quote=True.
    # Using quote=True ensures both double and single quotes are escaped.
    return html.escape(str(value), quote=True)


def render_tag(tag: str) -> str:
    """
    Render a single hashtag as a span element.

    Args:
        tag: The raw tag text to render.

    Returns:
        An HTML string like "<span>sanitized-tag</span>".
    """
    # Sanitize the tag content to prevent script injection via tag text.
    return f"<span>{sanitize(tag)}</span>"


def render_tags_html(tags: Iterable[str]) -> str:
    """
    Render an iterable of hashtags as space-separated span elements.

    Args:
        tags: An iterable of raw tag strings.

    Returns:
        A single HTML string consisting of multiple <span> elements
        separated by a single space. If no tags are provided, returns
        an empty string.
    """
    # (tags or []) safely handles cases where tags might be falsy
    # (e.g., None, empty list). Each tag is sanitized via render_tag.
    return " ".join(render_tag(tag) for tag in (tags or []))


def compose_post_html(
    poster_name: str,
    headline: str,
    text: str,
    hashtags: List[str],
) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    The function uses a predefined HTML template (USER_POST_FORMAT) and
    injects sanitized content for the poster's name, headline, main text,
    and associated tags.

    Args:
        poster_name: The name of the post's author.
        headline: The headline of the user's post.
        text: The primary text content of the user's post.
        hashtags: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the
        user's forum post.
    """
    # Sanitize user-provided fields before interpolation.
    safe_poster_name = sanitize(poster_name)
    safe_headline = sanitize(headline)
    safe_text = sanitize(text)

    # Render tags into a space-separated list of <span> elements.
    tags_html = render_tags_html(hashtags)

    # Interpolate sanitized content into the HTML template.
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html,
    )
