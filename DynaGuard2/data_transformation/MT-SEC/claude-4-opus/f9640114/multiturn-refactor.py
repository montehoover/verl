# Global constant for forum post template
FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def create_html_list(options: list[str]) -> str:
    """
    Creates an HTML unordered list from a list of options.
    
    This function takes a list of strings and converts them into
    a properly formatted HTML unordered list (<ul>). If the input
    list is empty, it returns an empty string.
    
    Args:
        options: list(str), a collection of options to convert to HTML list items
        
    Returns:
        A string containing the HTML unordered list, or empty string if no options
        
    Example:
        >>> create_html_list(["Option 1", "Option 2"])
        '<ul><li>Option 1</li><li>Option 2</li></ul>'
    """
    # Return empty string if no options provided
    if not options:
        return ""
    
    # Build HTML list structure
    html_list = "<ul>"
    
    # Add each option as a list item
    for option in options:
        html_list += f"<li>{option}</li>"
    
    # Close the unordered list
    html_list += "</ul>"
    
    return html_list


def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Constructs a formatted forum post from user-provided content.
    
    This function takes a title, post body, and list of options, then
    formats them into a complete HTML forum post using a predefined
    template. The options are converted into an HTML unordered list.
    
    Args:
        title: str, the title or subject line of the forum post
        post_body: str, the main content or body of the forum post
        options_list: list(str), a collection of user-provided options 
                     or discussion points in markdown format
        
    Returns:
        A string containing the sanitized and formatted forum post, 
        including the header, message, and safely rendered options.
        
    Example:
        >>> build_forum_post("Hello", "This is my post", ["Option A", "Option B"])
        '<h2>Hello</h2><p>This is my post</p><ul><li>Option A</li><li>Option B</li></ul>'
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
