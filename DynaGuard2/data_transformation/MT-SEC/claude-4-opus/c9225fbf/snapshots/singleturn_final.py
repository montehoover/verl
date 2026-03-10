USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def create_post_html(author_name, post_title, content, post_tags):
    """
    Constructs an HTML representation of a user's forum post.
    
    Args:
        author_name (str): The name of the post's author
        post_title (str): The headline of the user's post
        content (str): The primary text content of the user's post
        post_tags (list[str]): A collection of tags related to the post
    
    Returns:
        str: A string containing the XSS-protected HTML representation of the user's forum post
    """
    # Create HTML for tags - each tag wrapped in a <span> element
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in post_tags)
    
    # Use the provided template to format the post
    html_output = USER_POST_FORMAT.format(
        post_title=post_title,
        author_name=author_name,
        content=content,
        tags_html=tags_html
    )
    
    return html_output
