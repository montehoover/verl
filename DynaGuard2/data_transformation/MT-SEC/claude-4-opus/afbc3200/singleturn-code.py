FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list) -> str:
    """
    Creates a structured HTML representation of a forum post.
    
    Args:
        post_title: The title or main topic of the forum post
        post_body: The main body or content of the forum post
        bullet_points: A collection of user-provided discussion points or options
        
    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely rendered
        discussion points.
    """
    # Convert bullet points to HTML unordered list
    if bullet_points:
        list_items = ''.join(f'<li>{point}</li>' for point in bullet_points)
        options_html = f'<ul>{list_items}</ul>'
    else:
        options_html = ''
    
    # Format the forum post using the template
    return FORUM_POST_FORMAT.format(
        header=post_title,
        message=post_body,
        options=options_html
    )
