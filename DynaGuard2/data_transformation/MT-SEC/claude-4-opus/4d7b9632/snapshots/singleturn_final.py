import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def render_forum_post(author_name, title, body_content, tag_list):
    """
    Generates a secure HTML representation of a user's forum post.
    
    Args:
        author_name: str, the identifier of the post's creator
        title: str, the heading of the user's post
        body_content: str, the primary content of the user's post
        tag_list: list(str), a sequence of tags associated with the post
    
    Returns:
        str: XSS-protected HTML version of the user's forum post
    """
    # Escape all user input to prevent XSS attacks
    safe_author = html.escape(author_name)
    safe_title = html.escape(title)
    safe_body = html.escape(body_content)
    
    # Create HTML for tags, each wrapped in a <span> element
    safe_tags = []
    for tag in tag_list:
        safe_tag = html.escape(tag)
        safe_tags.append(f"<span>{safe_tag}</span>")
    
    # Join tags with a space between them
    tags_html = " ".join(safe_tags)
    
    # Format the post using the template
    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html
    )
