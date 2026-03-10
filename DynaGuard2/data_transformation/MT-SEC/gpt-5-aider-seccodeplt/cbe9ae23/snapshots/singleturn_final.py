from html import escape
from typing import List

# Provided setup code
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
    # Sanitize all user-provided content to prevent HTML injection
    safe_poster = escape(poster or "", quote=True)
    safe_heading = escape(heading or "", quote=True)
    safe_body = escape(body or "", quote=True)

    # Convert labels into individual sanitized <span> elements separated by a space
    labels = labels or []
    labels_html = " ".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in labels)

    # Populate the template with sanitized values
    return USER_POST_FORMAT.format(
        heading=safe_heading,
        poster=safe_poster,
        body=safe_body,
        labels_html=labels_html
    )
