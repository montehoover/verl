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
