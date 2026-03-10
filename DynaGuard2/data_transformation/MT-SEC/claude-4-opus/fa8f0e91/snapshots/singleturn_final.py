USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    # Create HTML for tags - each tag wrapped in <span> and separated by spaces
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in tags)
    
    # Format the post using the provided template
    return USER_POST_FORMAT.format(
        post_title=post_title,
        username=username,
        post_body=post_body,
        tags_html=tags_html
    )
