import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def _sanitize_input(text_input: str) -> str:
    """Escapes HTML special characters in a string."""
    return html.escape(text_input)

def _create_tags_html(hashtags: list[str]) -> str:
    """Creates an HTML string of span-enclosed, sanitized tags."""
    sanitized_tags = [_sanitize_input(tag) for tag in hashtags]
    tags_html_list = [f"<span>{tag}</span>" for tag in sanitized_tags]
    return "".join(tags_html_list)

def compose_post_html(poster_name: str, headline: str, text: str, hashtags: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post.

    Args:
        poster_name: The name of the post's author.
        headline: The headline of the user's post.
        text: The primary text content of the user's post.
        hashtags: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the user's forum post.
    """
    sanitized_headline = _sanitize_input(headline)
    sanitized_poster_name = _sanitize_input(poster_name)
    sanitized_text = _sanitize_input(text)
    
    tags_html_content = _create_tags_html(hashtags)

    return USER_POST_FORMAT.format(
        headline=sanitized_headline,
        poster_name=sanitized_poster_name,
        text=sanitized_text,
        tags_html=tags_html_content
    )
