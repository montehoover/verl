def detailed_post_json(author_id: str, post_heading: str, content: str) -> dict:
    """
    Generates a detailed dictionary structure for a blog post.

    Args:
        author_id: The ID of the post author.
        post_heading: The heading/title of the post.
        content: The main content of the post.

    Returns:
        A dictionary with author_id, title, and content.
    """
    return {
        "author_id": author_id,
        "title": post_heading,
        "content": content
    }
