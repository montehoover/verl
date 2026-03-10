import html

def create_post_html(headline: str, content: str) -> str:
    """
    Create an HTML snippet for a forum post.

    Args:
        headline: The post headline.
        content: The post content.

    Returns:
        An HTML string in the format: <h2>{headline}</h2><p>{content}</p>
    """
    return f"<h2>{headline}</h2><p>{content}</p>"

def format_discussion_points(points: list[str]) -> str:
    """
    Format a list of discussion points into an HTML unordered list.

    Args:
        points: A list of strings representing discussion points.

    Returns:
        A string of HTML with each point wrapped in <li> and enclosed in <ul>.
    """
    items = "".join(f"<li>{html.escape(point)}</li>" for point in points)
    return f"<ul>{items}</ul>"
