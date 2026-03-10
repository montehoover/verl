from html import escape

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def create_user_profile_html(user_name: str, user_bio: str) -> str:
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    return " ".join(f"<span>{escape(tag, quote=True)}</span>" for tag in tags)

def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: list[str]) -> str:
    safe_user_name = escape(user_name, quote=True)
    safe_post_heading = escape(post_heading, quote=True)
    safe_post_content = escape(post_content, quote=True)
    tags_html = generate_tags_html(post_tags)
    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        user_name=safe_user_name,
        post_content=safe_post_content,
        tags_html=tags_html
    )
