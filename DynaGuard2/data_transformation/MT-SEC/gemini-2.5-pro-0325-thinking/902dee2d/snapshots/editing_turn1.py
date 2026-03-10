def simple_formatted_post(headline: str, content: str) -> str:
    """
    Creates a basic markdown structure for a post.

    Args:
        headline: The headline of the post.
        content: The content of the post.

    Returns:
        A markdown formatted string.
    """
    return f"# {headline}\n\n{content}"
