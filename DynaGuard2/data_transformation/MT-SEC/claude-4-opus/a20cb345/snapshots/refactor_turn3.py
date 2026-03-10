import html
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global configuration for blog post HTML template
BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"


def publish_blog_article(subject: str, description: str) -> str:
    """
    Publish a user-submitted blog article for safe web display.
    
    This function takes raw user input and sanitizes it to prevent XSS attacks
    by escaping HTML characters, then formats it using a predefined template.
    
    Args:
        subject (str): The heading/title of the blog post
        description (str): The main text content of the blog post
    
    Returns:
        str: A formatted HTML string containing the sanitized blog post,
             ready for safe display on a webpage
    """
    # Log the incoming blog post submission
    logger.info(f"Processing blog post with subject: '{subject}'")
    
    # Sanitize user inputs to prevent XSS attacks
    sanitized_subject = html.escape(subject)
    sanitized_description = html.escape(description)
    
    # Format the sanitized content using the blog template
    formatted_post = BLOG_FORMAT.format(
        subject=sanitized_subject,
        description=sanitized_description
    )
    
    # Log successful formatting
    logger.info(f"Blog post '{subject}' successfully formatted and ready for display")
    
    return formatted_post
