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

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """
    Generates a structured HTML representation of a forum post.

    Args:
        topic: The title of the forum post.
        text_body: The main content of the forum post.
        points: A list of discussion points.

    Returns:
        A string containing the fully formatted HTML for the forum post.
    """
    # Note: The request implies generate_html_paragraph should be used for text_body.
    # However, FORUM_POST_FORMAT already includes <p> tags for the message.
    # If generate_html_paragraph were used, it would result in nested <p><p>...</p></p>.
    # Therefore, text_body is used directly for the 'message' placeholder.
    # If the intent was to use generate_html_paragraph, FORUM_POST_FORMAT should be
    # "<h2>{header}</h2>{message}{options}" and then message would be
    # generate_html_paragraph(text_body).
    # For now, adhering strictly to the provided FORUM_POST_FORMAT.

    formatted_points = format_discussion_points(points)
    return FORUM_POST_FORMAT.format(header=topic, message=text_body, options=formatted_points)
