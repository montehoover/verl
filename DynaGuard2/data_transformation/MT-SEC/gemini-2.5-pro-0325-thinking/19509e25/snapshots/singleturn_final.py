import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

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
    escaped_poster_name = html.escape(poster_name)
    escaped_headline = html.escape(headline)
    escaped_text = html.escape(text)

    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in hashtags]
    tags_html_content = " ".join(escaped_tags)

    return USER_POST_FORMAT.format(
        headline=escaped_headline,
        poster_name=escaped_poster_name,
        text=escaped_text,
        tags_html=tags_html_content
    )
