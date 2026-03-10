import html
from typing import List

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
