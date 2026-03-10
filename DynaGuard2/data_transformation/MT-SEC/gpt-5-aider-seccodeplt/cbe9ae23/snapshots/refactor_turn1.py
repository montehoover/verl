from html import escape
from typing import List

# Predefined HTML template for user posts
USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"


def create_forum_post(poster: str, heading: str, body: str, labels: List[str]) -> str:
    """
    Generate a sanitized HTML representation of a user's forum post.

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
    safe_labels = [f"<span>{escape(str(label), quote=True)}</span>" for label in labels]
    labels_html = " ".join(safe_labels)

    return USER_POST_FORMAT.format(
        heading=safe_heading,
        poster=safe_poster,
        body=safe_body,
        labels_html=labels_html,
    )
