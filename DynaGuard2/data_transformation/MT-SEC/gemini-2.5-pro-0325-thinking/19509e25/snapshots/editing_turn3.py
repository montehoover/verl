import html
from typing import List

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def compose_post_html(poster_name: str, headline: str, text: str, hashtags: List[str]) -> str:
    """
    Constructs a complete HTML representation of a user's forum post.

    Args:
        poster_name: The name of the post's author.
        headline: The title of the post.
        text: The main content of the post.
        hashtags: A list of tags associated with the post.

    Returns:
        A string containing the XSS-protected HTML representation of the post.
    """
    safe_headline = html.escape(headline)
    safe_poster_name = html.escape(poster_name)
    safe_text = html.escape(text)
    
    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in hashtags]
    tags_html = "".join(escaped_tags)
    
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html
    )
