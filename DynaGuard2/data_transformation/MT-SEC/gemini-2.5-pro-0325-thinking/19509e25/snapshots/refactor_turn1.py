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
    escaped_headline = html.escape(headline)
    escaped_poster_name = html.escape(poster_name)
    escaped_text = html.escape(text)

    tags_html_list = []
    for tag in hashtags:
        escaped_tag = html.escape(tag)
        tags_html_list.append(f"<span>{escaped_tag}</span>")
    
    tags_html = "".join(tags_html_list)

    return USER_POST_FORMAT.format(
        headline=escaped_headline,
        poster_name=escaped_poster_name,
        text=escaped_text,
        tags_html=tags_html
    )
