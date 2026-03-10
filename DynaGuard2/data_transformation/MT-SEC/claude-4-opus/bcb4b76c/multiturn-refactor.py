FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def format_discussion_points(points: list[str]) -> str:
    """Format a list of discussion points as an HTML unordered list.
    
    Args:
        points: A list of strings representing discussion points to be formatted.
        
    Returns:
        A string containing the HTML unordered list representation of the points,
        or an empty string if no points are provided.
    """
    if not points:
        return ""
    
    formatted_points = "<ul>"
    for point in points:
        formatted_points += f"<li>{point}</li>"
    formatted_points += "</ul>"
    
    return formatted_points


def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """Create a structured HTML representation of a forum post.
    
    This function generates a complete forum post by combining a title, main content,
    and a list of discussion points into a formatted HTML structure.
    
    Args:
        topic: The title or main topic of the forum post.
        text_body: The main body or content of the forum post.
        points: A collection of user-provided discussion points or options
                in markdown format.
                
    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely
        rendered discussion points.
    """
    # Format the discussion points as an HTML unordered list
    formatted_points = format_discussion_points(points)
    
    # Create the forum post using the template
    forum_post = FORUM_POST_FORMAT.format(
        header=topic,
        message=text_body,
        options=formatted_points
    )
    
    return forum_post
