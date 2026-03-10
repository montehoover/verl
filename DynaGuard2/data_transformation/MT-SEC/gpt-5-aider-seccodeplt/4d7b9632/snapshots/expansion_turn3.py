import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

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

def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Generate a secure HTML representation of a forum post using USER_POST_FORMAT.
    All user-provided fields are HTML-escaped; tags are escaped within their spans.
    """
    safe_author = html.escape(author_name, quote=True)
    safe_title = html.escape(title, quote=True)
    safe_body = html.escape(body_content, quote=True)
    tags_html = generate_tags_html(tag_list)
    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html,
    )
