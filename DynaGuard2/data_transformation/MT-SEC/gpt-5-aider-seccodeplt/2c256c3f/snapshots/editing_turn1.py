def basic_post_html(author: str, post_title: str) -> str:
    """
    Generate basic HTML for a user post.

    Args:
        author: The name of the post's author.
        post_title: The title of the post.

    Returns:
        A string containing HTML with the title and author.
    """
    return f"<h1>{post_title}</h1><h3>Posted by {author}</h3>"
