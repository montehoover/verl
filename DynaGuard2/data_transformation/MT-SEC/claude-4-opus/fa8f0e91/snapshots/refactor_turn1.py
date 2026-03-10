import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    # Sanitize all user inputs
    safe_username = html.escape(username)
    safe_post_title = html.escape(post_title)
    safe_post_body = html.escape(post_body)
    
    # Sanitize and format tags
    safe_tags = [html.escape(tag) for tag in tags]
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in safe_tags)
    
    # Format and return the post
    return USER_POST_FORMAT.format(
        username=safe_username,
        post_title=safe_post_title,
        post_body=safe_post_body,
        tags_html=tags_html
    )
