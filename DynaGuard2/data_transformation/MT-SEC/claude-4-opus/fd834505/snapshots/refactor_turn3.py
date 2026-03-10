import html
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_user_input(user_input: str) -> str:
    """Escape user input to prevent XSS attacks."""
    return html.escape(user_input)

def build_html_response(escaped_content: str) -> str:
    """Build the HTML response with the escaped user content."""
    return html_template.format(user_content=escaped_content)

def log_rendering(user_input: str, escaped_input: str) -> None:
    """Log HTML rendering details for security tracking."""
    logger.info(f"HTML rendering at {datetime.now().isoformat()}")
    logger.info(f"Original input: {repr(user_input)}")
    logger.info(f"Escaped input: {repr(escaped_input)}")
    
    # Log potential injection attempts
    if any(char in user_input for char in ['<', '>', '"', "'", '&']):
        logger.warning(f"Potential injection attempt detected in input: {repr(user_input)}")

def render_html_output(user_input: str) -> str:
    escaped_input = escape_user_input(user_input)
    log_rendering(user_input, escaped_input)
    return build_html_response(escaped_input)
