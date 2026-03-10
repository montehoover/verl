def format_user_message_with_title(username, title, message):
    return f"Title: {title}, User: {username}, Message: {message}"

def render_user_post(username, post_title, post_body, tags):
    USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"
    
    # Create HTML for tags
    tags_html = ' '.join([f'<span>{tag}</span>' for tag in tags])
    
    # Format and return the post
    return USER_POST_FORMAT.format(
        post_title=post_title,
        username=username,
        post_body=post_body,
        tags_html=tags_html
    )
