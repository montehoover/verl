def basic_forum_post(title: str, content: str):
    """
    Creates a basic HTML representation of a forum post and prints it.

    Args:
        title: The title of the post.
        content: The content of the post.
    """
    html_output = f"<h2>{title}</h2>\n<p>{content}</p>"
    print(html_output)
