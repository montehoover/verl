import html
from typing import List

# Pre-existing template for user posts
USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"


def compose_post_html(poster_name: str, headline: str, text: str, hashtags: List[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        poster_name: The name of the post's author.
        headline: The headline of the user's post.
        text: The primary text content of the user's post.
        hashtags: A collection of tags related to the post.

    Returns:
        A string containing the HTML representation of the user's forum post.
    """
    safe_poster_name = html.escape(str(poster_name), quote=True)
    safe_headline = html.escape(str(headline), quote=True)
    safe_text = html.escape(str(text), quote=True)

    tags_html = " ".join(
        f"<span>{html.escape(str(tag), quote=True)}</span>"
        for tag in (hashtags or [])
    )

    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html,
    )
