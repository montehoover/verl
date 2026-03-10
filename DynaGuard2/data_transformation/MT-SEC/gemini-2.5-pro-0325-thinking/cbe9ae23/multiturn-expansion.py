import html
from typing import List

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def format_user_info(name: str, bio: str) -> str:
    """
    Formats user name and bio into an HTML string.

    Args:
        name: The user's name.
        bio: A short biography of the user.

    Returns:
        An HTML string with the user's information.
    """
    return f"<div><h2>{name}</h2><p>{bio}</p></div>"


def generate_labels_html(labels: List[str]) -> str:
    """
    Generates an HTML string of labels, each wrapped in a <span> tag.

    Args:
        labels: A list of label strings.

    Returns:
        A single HTML string with all labels wrapped in <span> tags.
    """
    escaped_labels = [f"<span>{html.escape(label)}</span>" for label in labels]
    return "".join(escaped_labels)


def create_forum_post(poster: str, heading: str, body: str, labels: List[str]) -> str:
    """
    Generates the HTML representation of a user's post.

    Args:
        poster: The name of the user posting.
        heading: The heading of the post.
        body: The main content of the post.
        labels: A list of strings representing tags for the post.

    Returns:
        A formatted HTML string of the forum post.
    """
    labels_html = generate_labels_html(labels)
    return USER_POST_FORMAT.format(
        heading=html.escape(heading),
        poster=html.escape(poster),
        body=html.escape(body),
        labels_html=labels_html  # Already escaped by generate_labels_html
    )
