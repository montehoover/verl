def create_user_profile_html(username: str, bio: str) -> str:
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    return " ".join(f"<span>{tag}</span>" for tag in tags)

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    tags_html = generate_tags_html(tags)
    return USER_POST_FORMAT.format(
        post_title=post_title,
        username=username,
        post_body=post_body,
        tags_html=tags_html
    )
