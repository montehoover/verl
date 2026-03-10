from html import escape
from typing import List

# Predefined HTML template for user posts
USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"


def render_labels_html(labels: List[str]) -> str:
    """
    Pure function that returns the HTML string for the given labels.
    Each label is HTML-escaped and wrapped in a <span> element.
    """
    if not labels:
        return ""
    return " ".join(f"<span>{escape(str(label), quote=True)}</span>" for label in labels)


def format_user_post(
    poster: str,
    heading: str,
    body: str,
    labels_html: str,
    template: str = USER_POST_FORMAT,
) -> str:
    """
    Pure function that fills the post template with the provided (already sanitized) values.
    """
    return template.format(
        heading=heading,
        poster=poster,
        body=body,
        labels_html=labels_html,
    )


def create_forum_post(poster: str, heading: str, body: str, labels: List[str]) -> str:
    """
    Generate a sanitized HTML representation of a user's forum post.

    This function escapes all user-provided fields to prevent HTML injection,
    renders the labels as individual <span> elements, and then fills the global
    USER_POST_FORMAT template with the sanitized values.

    Args:
        poster: The author of the post.
        heading: The title of the user's post.
        body: The main content of the user's post.
        labels: A list of tags associated with the post.

    Returns:
        A string containing the sanitized HTML representation of the user's post.
    """
    safe_poster = escape(poster, quote=True)
    safe_heading = escape(heading, quote=True)
    safe_body = escape(body, quote=True)
    labels_html = render_labels_html(labels)

    return format_user_post(
        poster=safe_poster,
        heading=safe_heading,
        body=safe_body,
        labels_html=labels_html,
    )
