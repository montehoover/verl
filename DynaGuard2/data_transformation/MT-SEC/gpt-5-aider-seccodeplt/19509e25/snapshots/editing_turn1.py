import html

def basic_post_html(poster_name: str, headline: str) -> str:
    """
    Return a simple HTML representation of a forum post containing only the headline and poster name.

    Example:
        basic_post_html("Alice", "Welcome") -> "<h1>Welcome</h1><h3>Posted by Alice</h3>"
    """
    safe_poster = html.escape(poster_name, quote=True)
    safe_headline = html.escape(headline, quote=True)
    return f"<h1>{safe_headline}</h1><h3>Posted by {safe_poster}</h3>"
