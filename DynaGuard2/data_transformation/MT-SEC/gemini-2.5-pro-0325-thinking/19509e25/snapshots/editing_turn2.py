import html

def detailed_post_html(poster_name: str, headline: str, text: str) -> str:
    """
    Creates a detailed HTML representation of a user's post.

    Args:
        poster_name: The name of the post's author.
        headline: The title of the post.
        text: The main content of the post.

    Returns:
        A string containing the HTML representation of the post.
    """
    safe_headline = html.escape(headline)
    safe_poster_name = html.escape(poster_name)
    safe_text = html.escape(text)
    return f"<h1>{safe_headline}</h1><h3>Posted by {safe_poster_name}</h3><div>{safe_text}</div>"
