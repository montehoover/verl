FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title, post_body, options_list):
    """
    Constructs a formatted forum post from user-provided content.
    
    Args:
        title (str): The title or subject line of the forum post
        post_body (str): The main content or body of the forum post
        options_list (list[str]): A collection of user-provided options or discussion points
        
    Returns:
        str: A string containing the sanitized and formatted forum post,
             including the header, message, and safely rendered options
    """
    # Build the HTML unordered list from the options
    if options_list:
        list_items = ''.join(f'<li>{option}</li>' for option in options_list)
        formatted_options = f'<ul>{list_items}</ul>'
    else:
        formatted_options = ''
    
    # Format the forum post using the template
    return FORUM_POST_FORMAT.format(
        title=title,
        post_body=post_body,
        options_list=formatted_options
    )
