USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def build_post_html(user_name, post_heading, post_content, post_tags):
    """
    Generates an HTML representation of a user's post for a public forum.
    
    Args:
        user_name (str): The author of the post
        post_heading (str): The title of the user's post
        post_content (str): The main content of the user's post
        post_tags (list[str]): A list of tags associated with the post
    
    Returns:
        str: A string containing the sanitized HTML representation of the user's post
    """
    # Create span elements for each tag
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in post_tags)
    
    # Format the HTML using the template
    html = USER_POST_FORMAT.format(
        post_heading=post_heading,
        user_name=user_name,
        post_content=post_content,
        tags_html=tags_html
    )
    
    return html
