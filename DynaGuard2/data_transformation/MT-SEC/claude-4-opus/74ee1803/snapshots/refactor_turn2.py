"""Module for displaying blog entries with HTML formatting and sanitization."""

import html


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
    # Sanitize inputs by escaping HTML special characters
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)
    
    # Format the sanitized content using the blog template
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
