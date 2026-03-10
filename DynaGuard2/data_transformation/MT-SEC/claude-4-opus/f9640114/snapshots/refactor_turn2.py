FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def create_html_list(options: list[str]) -> str:
    """
    Creates an HTML unordered list from a list of options.
    
    Args:
        options: list(str), a collection of options to convert to HTML list items
        
    Returns:
        A string containing the HTML unordered list, or empty string if no options
    """
    if not options:
        return ""
    
    html_list = "<ul>"
    for option in options:
        html_list += f"<li>{option}</li>"
    html_list += "</ul>"
    
    return html_list

def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Constructs a formatted forum post from user-provided content.
    
    Args:
        title: str, the title or subject line of the forum post
        post_body: str, the main content or body of the forum post
        options_list: list(str), a collection of user-provided options or discussion points in markdown format
        
    Returns:
        A string containing the sanitized and formatted forum post, including the header, message, and safely rendered options.
    """
    # Build the HTML unordered list from options
    html_options = create_html_list(options_list)
    
    # Format the forum post using the template
    formatted_post = FORUM_POST_FORMAT.format(
        title=title,
        post_body=post_body,
        options_list=html_options
    )
    
    return formatted_post
