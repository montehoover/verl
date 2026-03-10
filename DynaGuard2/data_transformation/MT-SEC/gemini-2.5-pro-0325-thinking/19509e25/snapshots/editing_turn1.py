def basic_post_html(poster_name: str, headline: str) -> str:
    """
    Creates a basic HTML representation of a user's post.

    Args:
        poster_name: The name of the post's author.
        headline: The title of the post.

    Returns:
        A string containing the HTML representation of the post.
    """
    return f"<h1>{headline}</h1><h3>Posted by {poster_name}</h3>"
