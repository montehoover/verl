import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def create_user_html(poster_name: str, user_content: str) -> str:
    """
    Generates an HTML string for displaying user information.

    Args:
        poster_name: The name of the poster.
        user_content: The content posted by the user.

    Returns:
        An HTML string with the user's details.
    """
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for displaying tags.

    Args:
        tags: A list of tags (strings).

    Returns:
        A single string where each tag is enclosed in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{html.escape(tag)}</span>" for tag in tags)


def compose_post_html(poster_name: str, headline: str, text: str, hashtags: list[str]) -> str:
    """
    Constructs an XSS-protected HTML representation of a user's forum post.

    Args:
        poster_name: The name of the poster.
        headline: The headline of the post.
        text: The main text content of the post.
        hashtags: A list of hashtags associated with the post.

    Returns:
        A string containing the HTML representation of the user's forum post.
    """
    escaped_poster_name = html.escape(poster_name)
    escaped_headline = html.escape(headline)
    escaped_text = html.escape(text)
    
    tags_html = generate_tags_html(hashtags)
    
    return USER_POST_FORMAT.format(
        headline=escaped_headline,
        poster_name=escaped_poster_name,
        text=escaped_text,
        tags_html=tags_html
    )
