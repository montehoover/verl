from html import escape

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    safe_username = escape(username, quote=True)
    safe_title = escape(post_title, quote=True)
    safe_body = escape(post_body, quote=True).replace("\n", "<br>")
    safe_tags = [escape(str(tag), quote=True) for tag in (tags or [])]
    tags_html = " ".join(f"<span>{t}</span>" for t in safe_tags)

    return USER_POST_FORMAT.format(
        post_title=safe_title,
        username=safe_username,
        post_body=safe_body,
        tags_html=tags_html,
    )
