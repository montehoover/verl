import html
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)

# HTML template with placeholder for user content
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"


def escape_html_content(content: str) -> str:
    """
    Escape HTML special characters in the content to prevent injection attacks.
    
    This function ensures that any HTML special characters (like <, >, &, etc.)
    are converted to their corresponding HTML entities, preventing malicious
    HTML or JavaScript injection.
    
    Args:
        content: The raw content to escape. Can contain any text including
                HTML special characters.
        
    Returns:
        The escaped content safe for HTML insertion. All special characters
        will be converted to HTML entities.
        
    Example:
        >>> escape_html_content("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;'
    """
    # Log the escaping operation for security auditing
    logger.debug(f"Escaping HTML content of length {len(content)}")
    
    # Use html.escape to convert special characters to HTML entities
    escaped = html.escape(content)
    
    logger.debug("HTML content successfully escaped")
    return escaped


def insert_content_into_template(template: str, escaped_content: str) -> str:
    """
    Insert pre-escaped content into the HTML template.
    
    This function performs the string formatting operation to insert the
    escaped user content into the designated placeholder within the HTML
    template.
    
    Args:
        template: The HTML template string containing a {user_content}
                 placeholder where the content should be inserted.
        escaped_content: The pre-escaped content to insert. This should
                        already be HTML-safe.
        
    Returns:
        The complete HTML string with content inserted at the placeholder
        position.
        
    Note:
        This function assumes the content has already been escaped and is
        safe for HTML insertion.
    """
    # Log template processing
    logger.debug("Inserting content into HTML template")
    
    # Format the template with the escaped content
    formatted_html = template.format(user_content=escaped_content)
    
    logger.debug(f"Generated HTML response of length {len(formatted_html)}")
    return formatted_html


def build_html_output(client_message: str) -> str:
    """
    Create a secure HTML response by safely embedding user content into a template.
    
    This is the main orchestrator function that coordinates the HTML generation
    process. It ensures user content is properly escaped before insertion into
    the HTML template, preventing XSS attacks and other HTML injection issues.
    
    Args:
        client_message: The content contributed by a user which will be
                       inserted into the HTML template. This can be any
                       string value, including potentially malicious content.
        
    Returns:
        The resulting HTML string that includes the user's content safely
        embedded into it. The output is a complete HTML document ready
        for rendering.
        
    Example:
        >>> build_html_output("Hello, World!")
        '<html><body><h1>Welcome!</h1><p>Hello, World!</p></body></html>'
        
        >>> build_html_output("<b>Bold text</b>")
        '<html><body><h1>Welcome!</h1><p>&lt;b&gt;Bold text&lt;/b&gt;</p></body></html>'
    """
    # Log the start of HTML generation
    logger.info(f"Building HTML output for client message of length {len(client_message)}")
    
    try:
        # Step 1: Escape the user content to prevent HTML injection
        escaped_content = escape_html_content(client_message)
        
        # Step 2: Insert the escaped content into the template
        html_output = insert_content_into_template(html_template, escaped_content)
        
        # Log successful completion
        logger.info("HTML output successfully generated")
        
        return html_output
        
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Error generating HTML output: {str(e)}", exc_info=True)
        raise
