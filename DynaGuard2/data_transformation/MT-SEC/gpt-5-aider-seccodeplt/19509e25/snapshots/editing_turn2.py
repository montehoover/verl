import html

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
