import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def sanitize_input(text):
    """Escape HTML special characters to prevent XSS attacks."""
    return html.escape(text)

def create_tags_html(hashtags):
    """Create HTML for tags, each wrapped in a span element."""
    return ''.join(f'<span>{sanitize_input(tag)}</span>' for tag in hashtags)

def compose_post_html(poster_name, headline, text, hashtags):
    # Sanitize all user inputs
    safe_poster_name = sanitize_input(poster_name)
    safe_headline = sanitize_input(headline)
    safe_text = sanitize_input(text)
    
    # Create HTML for tags
    tags_html = create_tags_html(hashtags)
    
    # Format and return the complete HTML
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html
    )
