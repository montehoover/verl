def basic_post_json(author_id: str, post_heading: str) -> dict:
    """
    Generates a basic dictionary structure for a blog post.

    Args:
        author_id: The ID of the post author.
        post_heading: The heading/title of the post.

    Returns:
        A dictionary with author_id, title, and default empty content.
    """
    return {
        "author_id": author_id,
        "title": post_heading,
        "content": ""
    }
