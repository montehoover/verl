def generate_html_paragraph(text_content: str) -> str:
    """
    Wraps the given text content in HTML <p> tags.

    Args:
        text_content: The string to be wrapped.

    Returns:
        A string representing an HTML paragraph.
    """
    return f"<p>{text_content}</p>"


def format_discussion_points(discussion_points: list[str]) -> str:
    """
    Formats a list of discussion points as an HTML unordered list.

    Args:
        discussion_points: A list of strings, where each string is a discussion point.

    Returns:
        A string representing an HTML unordered list (<ul>) with each point
        as a list item (<li>).
    """
    if not discussion_points:
        return "<ul></ul>"
    list_items = "".join([f"<li>{point}</li>" for point in discussion_points])
    return f"<ul>{list_items}</ul>"
