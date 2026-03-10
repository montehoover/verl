USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"


def sanitize(text):
    """Sanitize text to prevent XSS attacks.
    
    Args:
        text (str): The text to sanitize.
        
    Returns:
        str: The sanitized text with HTML special characters escaped.
    """
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')


def generate_tags_html(tags):
    """Generate HTML for tags as separate span elements.
    
    Args:
        tags (list): A list of tag strings to convert to HTML.
        
    Returns:
        str: HTML string with each tag wrapped in a span element.
    """
    return ''.join([f'<span>{sanitize(tag)}</span>' for tag in tags])


def build_post_html(user_name, post_heading, post_content, post_tags):
    """Generate an HTML representation of a user's post for a public forum.
    
    Args:
        user_name (str): The author of the post.
        post_heading (str): The title of the user's post.
        post_content (str): The main content of the user's post.
        post_tags (list): A list of tags associated with the post.
        
    Returns:
        str: A string containing the sanitized HTML representation of the user's post.
    """
    # Sanitize all text inputs
    safe_user_name = sanitize(user_name)
    safe_post_heading = sanitize(post_heading)
    safe_post_content = sanitize(post_content)
    
    # Generate tags HTML
    tags_html = generate_tags_html(post_tags)
    
    # Format and return the HTML
    return USER_POST_FORMAT.format(
        user_name=safe_user_name,
        post_heading=safe_post_heading,
        post_content=safe_post_content,
        tags_html=tags_html
    )
