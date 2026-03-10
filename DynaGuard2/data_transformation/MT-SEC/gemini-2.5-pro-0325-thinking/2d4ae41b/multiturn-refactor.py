import html
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def _escape_html_content(content: str) -> str:
    """Escapes HTML special characters in a string."""
    return html.escape(content)

def _insert_content_into_template(template: str, content: str) -> str:
    """Inserts content into an HTML template."""
    return template.format(user_content=content)

def render_html_page(visitor_message: str) -> str:
    """
    Creates an HTML response by inserting user content into a given HTML structure.

    The generated response will display user-provided information within a predefined format.

    Args:
        visitor_message: The content contributed by a user which will be
                         inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely embedded into it.
    
    Raises:
        No explicit exceptions are handled, though improper HTML escaping could
        potentially lead to undesired outcomes.
    """
    logger.info(f"Rendering HTML page with visitor_message: '{visitor_message}'")
    safe_user_content = _escape_html_content(visitor_message)
    html_output = _insert_content_into_template(html_template, safe_user_content)
    logger.info(f"Generated HTML output: '{html_output}'")
    return html_output
