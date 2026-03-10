def detailed_post_json(author_id: str, post_heading: str, content: str) -> dict:
    """
    Generate a JSON-like dict for a blog post containing the author, title, and content.

    Args:
        author_id (str): The author's identifier.
        post_heading (str): The post's title.
        content (str): The post's content.

    Returns:
        dict: A dictionary with keys "author_id", "title", and "content".
    """
    if not isinstance(author_id, str):
        raise TypeError("author_id must be a string")
    if not isinstance(post_heading, str):
        raise TypeError("post_heading must be a string")
    if not isinstance(content, str):
        raise TypeError("content must be a string")

    return {
        "author_id": author_id,
        "title": post_heading,
        "content": content
    }
