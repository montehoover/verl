import html

def create_user_html(author_name: str, user_bio: str) -> str:
    """
    Return an HTML string with the author's name and bio wrapped in a div.
    Format: <div><h2>{author_name}</h2><p>{user_bio}</p></div>
    """
    return f"<div><h2>{author_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    """
    Return a single HTML string where each tag is wrapped in a <span> element,
    separated by spaces. Tag content is HTML-escaped for safety.
    """
    return " ".join(f"<span>{html.escape(tag, quote=True)}</span>" for tag in tags)
