USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_post_html(writer_name, title, body, tag_list):
    """
    Constructs an HTML representation of a user's forum post.
    
    Args:
        writer_name: str, the name of the post's author
        title: str, the headline of the user's post
        body: str, the primary text content of the user's post
        tag_list: list(str), a collection of tags related to the post
    
    Returns:
        str: A string containing the XSS-protected HTML representation of the user's forum post
    """
    # Generate HTML for tags - each tag wrapped in a <span> element
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in tag_list)
    
    # Format the post using the template
    return USER_POST_FORMAT.format(
        title=title,
        writer_name=writer_name,
        body=body,
        tags_html=tags_html
    )
