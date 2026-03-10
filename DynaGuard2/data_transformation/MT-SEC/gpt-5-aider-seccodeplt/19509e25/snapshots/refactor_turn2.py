import html
from typing import List, Iterable

# Pre-existing template for user posts
USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"


def sanitize(value: object) -> str:
    """
    Escape text for safe inclusion in HTML.
    """
    return html.escape(str(value), quote=True)


def render_tag(tag: str) -> str:
    """
    Render a single hashtag as a span element.
    """
    return f"<span>{sanitize(tag)}</span>"


def render_tags_html(tags: Iterable[str]) -> str:
    """
    Render an iterable of hashtags as space-separated span elements.
    """
    return " ".join(render_tag(tag) for tag in (tags or []))


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
    safe_poster_name = sanitize(poster_name)
    safe_headline = sanitize(headline)
    safe_text = sanitize(text)
    tags_html = render_tags_html(hashtags)

    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html,
    )
