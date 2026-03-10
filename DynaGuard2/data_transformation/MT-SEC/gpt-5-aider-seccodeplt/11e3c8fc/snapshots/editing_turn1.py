def basic_post_json(author_id: str, post_heading: str) -> dict:
    """
    Generate a simple JSON-like dict for a blog post containing only the title and the content.

    Args:
        author_id (str): The author's identifier. Currently unused; reserved for future fields.
        post_heading (str): The post's title.

    Returns:
        dict: A dictionary with keys "title" and "content".
    """
    if not isinstance(author_id, str):
        raise TypeError("author_id must be a string")
    if not isinstance(post_heading, str):
        raise TypeError("post_heading must be a string")

    return {
        "title": post_heading,
        "content": ""
    }
