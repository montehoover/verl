import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def detailed_post_html(poster_name: str, headline: str, text: str) -> str:
    """
    Return a simple HTML representation of a forum post containing the headline,
    poster name, and main content.

    Example:
        detailed_post_html("Alice", "Welcome", "Hello world!") -> "<h1>Welcome</h1><h3>Posted by Alice</h3><div>Hello world!</div>"
    """
    safe_poster = html.escape(poster_name, quote=True)
    safe_headline = html.escape(headline, quote=True)
    safe_text = html.escape(text, quote=True)
    return f"<h1>{safe_headline}</h1><h3>Posted by {safe_poster}</h3><div>{safe_text}</div>"

def compose_post_html(poster_name: str, headline: str, text: str, hashtags: list[str]) -> str:
    """
    Construct a complete HTML representation of a user's forum post using the global USER_POST_FORMAT.
    Applies HTML escaping to all user-supplied fields to prevent XSS.
    """
    safe_poster = html.escape(poster_name, quote=True)
    safe_headline = html.escape(headline, quote=True)
    safe_text = html.escape(text, quote=True)
    safe_tags = (html.escape(tag, quote=True) for tag in hashtags)
    tags_html = "".join(f"<span>{tag}</span>" for tag in safe_tags)
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster,
        text=safe_text,
        tags_html=tags_html,
    )
