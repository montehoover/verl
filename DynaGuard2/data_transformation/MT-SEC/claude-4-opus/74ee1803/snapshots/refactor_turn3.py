"""Module for displaying blog entries with HTML formatting and sanitization."""

import html
import logging


# HTML template for formatting blog posts
BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"


def display_blog_entry(headline: str, text: str) -> str:
    """
    Format and sanitize a blog entry for safe HTML display.
    
    This function takes a blog post's title and content, sanitizes them to prevent
    XSS attacks by escaping HTML special characters, and formats them using a
    predefined HTML template.
    
    Args:
        headline: The title or headline of the blog post.
        text: The main content or body text of the blog post.
    
    Returns:
        A string containing the sanitized and HTML-formatted blog post,
        ready for safe display on a webpage.
    
    Example:
        >>> display_blog_entry("My Blog", "Hello <script>alert('XSS')</script>")
        '<h2>My Blog</h2><div>Hello &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;</div>'
    """
    # Initialize logger for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler with a simple format if logger has no handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Log the input parameters
    logger.info(f"Processing blog entry - Headline: '{headline}', Text length: {len(text)} characters")
    
    # Sanitize inputs by escaping HTML special characters
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)
    
    # Log sanitization results
    logger.info(f"Sanitized headline: '{sanitized_headline}'")
    logger.info(f"Sanitized text preview: '{sanitized_text[:50]}...' (first 50 chars)")
    
    # Format the sanitized content using the blog template
    formatted_html = BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
    
    # Log the final formatted HTML
    logger.info(f"Generated HTML length: {len(formatted_html)} characters")
    logger.info(f"HTML preview: '{formatted_html[:100]}...' (first 100 chars)")
    
    return formatted_html
