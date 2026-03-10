def create_post_html(post_title: str, post_body: str) -> str:
    """
    Generates an HTML string for a forum post.

    Args:
        post_title: The title of the post.
        post_body: The body content of the post.

    Returns:
        An HTML string formatted with the post title as an h2 tag
        and the post body as a p tag.
    """
    return f"<h2>{post_title}</h2><p>{post_body}</p>"


def format_discussion_points(discussion_points: list[str]) -> str:
    """
    Formats a list of discussion points as an HTML unordered list.

    Args:
        discussion_points: A list of strings, where each string is a discussion point.

    Returns:
        An HTML string representing an unordered list of the discussion points.
    """
    if not discussion_points:
        return "<ul></ul>"
    list_items = "".join([f"<li>{point}</li>" for point in discussion_points])
    return f"<ul>{list_items}</ul>"
