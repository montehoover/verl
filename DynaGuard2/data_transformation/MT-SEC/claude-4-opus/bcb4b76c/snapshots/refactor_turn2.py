FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def format_discussion_points(points: list[str]) -> str:
    """Format a list of discussion points as an HTML unordered list."""
    if not points:
        return ""
    
    formatted_points = "<ul>"
    for point in points:
        formatted_points += f"<li>{point}</li>"
    formatted_points += "</ul>"
    
    return formatted_points

def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    # Format the discussion points as an HTML unordered list
    formatted_points = format_discussion_points(points)
    
    # Create the forum post using the template
    forum_post = FORUM_POST_FORMAT.format(
        header=topic,
        message=text_body,
        options=formatted_points
    )
    
    return forum_post
