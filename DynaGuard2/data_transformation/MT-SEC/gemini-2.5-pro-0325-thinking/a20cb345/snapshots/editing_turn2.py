def format_blog_post(title: str, content: str) -> str:
    """Formats the blog post title and content for display in markdown.

    Args:
        title: The title of the blog post.
        content: The content of the blog post.

    Returns:
        A string formatted as '# {title}\n{content}'.
    """
    return f"# {title}\n{content}"
