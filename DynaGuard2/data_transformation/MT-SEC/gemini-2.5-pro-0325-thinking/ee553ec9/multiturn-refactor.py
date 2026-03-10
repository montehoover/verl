import html
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# HTML template string with a placeholder for user content.
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def _insert_content_into_template(template: str, content: str) -> str:
    """
    Inserts provided content into a given HTML template.

    This is a helper function designed to encapsulate the logic of formatting
    an HTML string with specific content. It replaces a designated placeholder
    within the template with the provided content.

    Args:
        template: The HTML template string. This string should contain
                  a placeholder (e.g., '{user_content}') that will be
                  replaced by the 'content' argument.
        content: The string content to be inserted into the template.
                 It is assumed that this content is already sanitized
                 if necessary (e.g., HTML escaped) before being passed
                 to this function.

    Returns:
        A new string representing the HTML template with the content
        inserted.
    """
    # Use the format method to replace the placeholder with the actual content.
    return template.format(user_content=content)

def build_html_output(client_message: str) -> str:
    """
    Generates a complete HTML page string with user-provided content safely embedded.

    This function takes a message from a client, sanitizes it to prevent
    Cross-Site Scripting (XSS) by escaping HTML special characters, and then
    inserts this sanitized message into a predefined HTML template.
    Logging is performed at the beginning and end of the process.

    Args:
        client_message: The raw string message provided by the user. This
                        message will be HTML-escaped before insertion.

    Returns:
        A string containing the full HTML page with the user's message
        safely embedded.
    """
    logging.info(f"Starting HTML output generation for client message: '{client_message[:30]}...'")

    # Sanitize the client's message by escaping HTML special characters.
    # This is a crucial step to prevent XSS attacks.
    safe_client_message = html.escape(client_message)
    logging.info("Client message has been HTML escaped.")

    # Insert the sanitized message into the HTML template.
    html_output = _insert_content_into_template(html_template, safe_client_message)
    logging.info("HTML output generated successfully.")

    return html_output
