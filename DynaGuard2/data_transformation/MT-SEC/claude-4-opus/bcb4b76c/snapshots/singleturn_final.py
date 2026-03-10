FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_forum_post(topic, text_body, points):
    """
    Creates a structured HTML representation of a forum post.
    
    Args:
        topic (str): The title or main topic of the forum post
        text_body (str): The main body or content of the forum post
        points (list[str]): A collection of user-provided discussion points or options
        
    Returns:
        str: A string containing the fully formatted HTML representation of the forum post
    """
    # Create the unordered list HTML from the points
    if points:
        list_items = ''.join(f'<li>{point}</li>' for point in points)
        options_html = f'<ul>{list_items}</ul>'
    else:
        options_html = ''
    
    # Format the complete forum post using the template
    forum_post = FORUM_POST_FORMAT.format(
        header=topic,
        message=text_body,
        options=options_html
    )
    
    return forum_post
