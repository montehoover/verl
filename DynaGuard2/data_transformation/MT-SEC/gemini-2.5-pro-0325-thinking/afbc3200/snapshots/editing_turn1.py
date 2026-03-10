def basic_forum_post(post_title: str, post_body: str) -> str:
    """
    Creates a simple XML representation of a forum post.

    Args:
        post_title: The title of the post.
        post_body: The body content of the post.

    Returns:
        A string with the title in a <title> tag and the body in a <content> tag.
    """
    return f"<post>\n  <title>{post_title}</title>\n  <content>{post_body}</content>\n</post>"
