import html
import logging


BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"  # HTML template for blog posts


def display_blog_entry(headline: str, text: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.

    The function first sanitizes the input headline and text to prevent
    cross-site scripting (XSS) vulnerabilities by escaping HTML special
    characters. Then, it formats the sanitized content using a predefined
    HTML template.

    Args:
        headline: The headline or title of the blog post.
        text: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post,
        safe for display on a webpage.
    """
    # Initialize logging configuration if not already configured
    # This basicConfig will only have an effect the first time it's called.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Received blog entry formatting request.")
    logger.info(f"Input headline: '{headline}'")
    logger.info(f"Input text: '{text[:100]}...' (truncated if long)") # Log a snippet of text

    # Sanitize inputs to prevent XSS by escaping special HTML characters.
    # This is crucial for security to avoid rendering malicious HTML/script content.
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)
    logger.info(f"Sanitized headline: '{sanitized_headline}'")
    logger.info(f"Sanitized text: '{sanitized_text[:100]}...' (truncated if long)")

    # Format the sanitized content into the blog post HTML structure.
    formatted_html = BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
    logger.info(f"Formatted HTML: '{formatted_html[:100]}...' (truncated if long)")

    return formatted_html
