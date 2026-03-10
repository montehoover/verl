import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def create_user_profile_html(username: str, bio: str) -> str:
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    return " ".join(f"<span>{html.escape(tag, quote=True)}</span>" for tag in tags)

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    safe_author_id = html.escape(author_id, quote=True)
    safe_post_heading = html.escape(post_heading, quote=True)
    safe_content = html.escape(content, quote=True)
    label_html = generate_tags_html(labels)
    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        author_id=safe_author_id,
        content=safe_content,
        label_html=label_html,
    )
