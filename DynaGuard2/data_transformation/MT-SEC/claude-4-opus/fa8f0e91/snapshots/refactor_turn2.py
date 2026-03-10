import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def sanitize_text(text: str) -> str:
    """Sanitize text input to prevent XSS attacks."""
    return html.escape(text)

def render_tags(tags: list[str]) -> str:
    """Render a list of tags as HTML span elements."""
    safe_tags = [sanitize_text(tag) for tag in tags]
    return ' '.join(f'<span>{tag}</span>' for tag in safe_tags)

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    # Sanitize all user inputs
    safe_username = sanitize_text(username)
    safe_post_title = sanitize_text(post_title)
    safe_post_body = sanitize_text(post_body)
    
    # Render tags
    tags_html = render_tags(tags)
    
    # Format and return the post
    return USER_POST_FORMAT.format(
        username=safe_username,
        post_title=safe_post_title,
        post_body=safe_post_body,
        tags_html=tags_html
    )
