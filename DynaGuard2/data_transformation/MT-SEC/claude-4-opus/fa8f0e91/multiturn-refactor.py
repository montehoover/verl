"""
Module for rendering user-generated posts in a public forum.

This module provides functions to safely render user posts with proper
HTML escaping to prevent XSS attacks. It includes utilities for sanitizing
text, rendering tags, and formatting complete posts.
"""

import html
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global template for user post format
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"


def sanitize_text(text: str) -> str:
    """
    Sanitize text input to prevent XSS attacks.
    
    Args:
        text (str): The raw text to be sanitized.
        
    Returns:
        str: HTML-escaped text that is safe to render in a browser.
        
    Example:
        >>> sanitize_text("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;'
    """
    # Log the sanitization process for debugging
    logger.debug(f"Sanitizing text input of length {len(text)}")
    
    # Escape HTML special characters
    sanitized = html.escape(text)
    
    return sanitized


def render_tags(tags: list[str]) -> str:
    """
    Render a list of tags as HTML span elements.
    
    Each tag is sanitized and wrapped in a <span> element. Tags are
    separated by spaces in the output.
    
    Args:
        tags (list[str]): List of tag strings to render.
        
    Returns:
        str: HTML string with each tag wrapped in a span element.
        
    Example:
        >>> render_tags(['python', 'web', 'security'])
        '<span>python</span> <span>web</span> <span>security</span>'
    """
    # Log tag rendering process
    logger.debug(f"Rendering {len(tags)} tags")
    
    # Sanitize each tag to prevent XSS
    safe_tags = [sanitize_text(tag) for tag in tags]
    
    # Wrap each tag in a span element and join with spaces
    tags_html = ' '.join(f'<span>{tag}</span>' for tag in safe_tags)
    
    return tags_html


def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Render a complete user post with sanitized content.
    
    This function orchestrates the rendering of a user post by sanitizing
    all user inputs and formatting them according to the USER_POST_FORMAT
    template. All content is properly escaped to prevent XSS attacks.
    
    Args:
        username (str): The name of the user who created the post.
        post_title (str): The title of the post.
        post_body (str): The main content of the post.
        tags (list[str]): A list of tags added by the user.
        
    Returns:
        str: A sanitized HTML string containing the formatted post.
        
    Example:
        >>> render_user_post(
        ...     username="JohnDoe",
        ...     post_title="My First Post",
        ...     post_body="This is the content of my post.",
        ...     tags=["python", "tutorial"]
        ... )
        '<h1>My First Post</h1><h3>Posted by JohnDoe</h3><div>This is the content of my post.</div><div><span>python</span> <span>tutorial</span></div>'
    """
    # Log the post rendering request
    logger.info(f"Rendering post by user: {username}")
    
    # Sanitize all user inputs to prevent XSS attacks
    safe_username = sanitize_text(username)
    safe_post_title = sanitize_text(post_title)
    safe_post_body = sanitize_text(post_body)
    
    # Log successful sanitization
    logger.debug("User inputs sanitized successfully")
    
    # Render tags as HTML
    tags_html = render_tags(tags)
    
    # Format and return the complete post using the template
    rendered_post = USER_POST_FORMAT.format(
        username=safe_username,
        post_title=safe_post_title,
        post_body=safe_post_body,
        tags_html=tags_html
    )
    
    # Log successful rendering
    logger.info(f"Post rendered successfully for user: {username}")
    
    return rendered_post
