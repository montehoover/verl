from typing import List

def format_post_with_points(title: str, body: str, points: List[str]) -> str:
    """
    Formats a post with a title, body, and a list of discussion points.

    Args:
        title: The title of the post.
        body: The body content of the post.
        points: A list of strings, where each string is a discussion point.

    Returns:
        A formatted string representing the post with bullet points.
    """
    points_formatted = "\n".join([f"- {point}" for point in points])
    return f"Title: {title}, Body: {body}, Points:\n{points_formatted}"
