def forum_post_with_points(title: str, content: str, points: list[str]):
    """
    Creates a basic HTML representation of a forum post with discussion points and prints it.

    Args:
        title: The title of the post.
        content: The content of the post.
        points: A list of discussion points (strings).
    """
    html_output = f"<h2>{title}</h2>\n<p>{content}</p>\n"
    if points:
        html_output += "Discussion Points:\n"
        for point in points:
            html_output += f"- {point}\n"
    print(html_output.strip())
