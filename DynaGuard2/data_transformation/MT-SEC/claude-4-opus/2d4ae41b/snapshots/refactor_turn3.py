import html
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_user_content(content: str) -> str:
    """Safely escape HTML special characters in user content."""
    return html.escape(content)

def insert_content_into_template(template: str, escaped_content: str) -> str:
    """Insert escaped content into HTML template."""
    return template.format(user_content=escaped_content)

def render_html_page(visitor_message: str) -> str:
    logger.info(f"Starting HTML rendering for visitor message: {visitor_message}")
    
    escaped_content = escape_user_content(visitor_message)
    logger.debug(f"Escaped content: {escaped_content}")
    
    final_html = insert_content_into_template(html_template, escaped_content)
    logger.info(f"HTML rendering complete. Output: {final_html}")
    
    return final_html
