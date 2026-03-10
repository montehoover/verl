import html


# Global HTML template for formatting user posts
USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"


def sanitize_input(text):
    """
    Escape HTML special characters to prevent XSS attacks.
    
    Args:
        text (str): The raw text input that may contain HTML special characters.
        
    Returns:
        str: The escaped text safe for HTML rendering.
    """
    return html.escape(text)


def create_tags_html(hashtags):
    """
    Create HTML for tags, each wrapped in a span element.
    
    Args:
        hashtags (list of str): A collection of tag strings to be rendered.
        
    Returns:
        str: HTML string with each tag wrapped in a <span> element.
    """
    # Generate a span element for each tag, with sanitized content
    return ''.join(f'<span>{sanitize_input(tag)}</span>' for tag in hashtags)


def compose_post_html(poster_name, headline, text, hashtags):
    """
    Construct an HTML representation of a user's forum post.
    
    This function takes the various components of a forum post and combines them
    into a formatted HTML string, ensuring all user inputs are properly sanitized
    to prevent XSS attacks.
    
    Args:
        poster_name (str): The name of the post's author.
        headline (str): The headline of the user's post.
        text (str): The primary text content of the user's post.
        hashtags (list of str): A collection of tags related to the post.
        
    Returns:
        str: The XSS-protected HTML representation of the user's forum post.
    """
    # Sanitize all user inputs to prevent XSS attacks
    safe_poster_name = sanitize_input(poster_name)
    safe_headline = sanitize_input(headline)
    safe_text = sanitize_input(text)
    
    # Create HTML representation for all tags
    tags_html = create_tags_html(hashtags)
    
    # Format and return the complete HTML using the global template
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html
    )
