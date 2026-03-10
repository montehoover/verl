def create_post_html(headline: str, content: str) -> str:
    """
    Generates an HTML string for a forum post.

    Args:
        headline: The headline of the post.
        content: The main content of the post.

    Returns:
        An HTML string formatted as a header and a paragraph.
    """
    return f"<h2>{headline}</h2><p>{content}</p>"


def format_discussion_points(points: list[str]) -> str:
    """
    Formats a list of discussion points into an HTML unordered list.

    Args:
        points: A list of strings, where each string is a discussion point.

    Returns:
        An HTML string representing an unordered list of the discussion points.
    """
    if not points:
        return ""
    list_items = "".join([f"<li>{point}</li>" for point in points])
    return f"<ul>{list_items}</ul>"
