"""
Utilities for building HTML output by safely inserting user content into
a predefined HTML template. Includes logging to monitor the response
generation process.
"""

import logging
from html import escape

# Module-level logger for monitoring HTML response generation.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Predefined HTML template with a placeholder for user content.
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"


def sanitize_user_content(client_message: str) -> str:
    """
    Return an HTML-escaped representation of user-provided content.

    This function performs HTML escaping to prevent user-supplied text
    from being interpreted as markup, reducing the risk of XSS if the
    generated HTML is rendered by a browser.

    Args:
        client_message: Raw text provided by the user.

    Returns:
        A string with HTML entities escaped, safe to insert into the template.
    """
    logger.debug(
        "Sanitizing user content (length=%d)", len(client_message)
    )

    # Escape user content to ensure it is treated as text, not HTML.
    safe_user_content = escape(client_message, quote=True)

    logger.debug(
        "Sanitized content length=%d", len(safe_user_content)
    )
    return safe_user_content


def insert_user_content_into_template(template: str, user_content: str) -> str:
    """
    Insert sanitized user content into the provided HTML template.

    The template must contain the placeholder '{user_content}' which will be
    replaced by the sanitized content.

    Args:
        template: The HTML template containing the '{user_content}' placeholder.
        user_content: The sanitized user content to be inserted.

    Returns:
        A complete HTML string with the user content embedded.
    """
    logger.debug(
        "Inserting user content into template "
        "(template_length=%d, content_length=%d)",
        len(template),
        len(user_content),
    )

    # Perform a simple placeholder replacement using str.format.
    # Assumes the template contains '{user_content}'.
    html = template.format(user_content=user_content)

    logger.debug("Generated HTML length=%d", len(html))
    return html


def build_html_output(client_message: str) -> str:
    """
    Build a complete HTML response string including user content.

    This function orchestrates the full process:
    - Sanitize the raw user message to make it safe for HTML.
    - Insert the sanitized content into the predefined HTML template.

    Args:
        client_message: The content contributed by a user which will be
            inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely
        embedded into it.

    Note:
        No explicit exceptions are handled here. Any errors raised by
        string formatting or escaping will propagate to the caller.
    """
    logger.info("Building HTML output")

    # Sanitize user input to prevent injection of HTML/JS.
    safe_user_content = sanitize_user_content(client_message)

    # Merge the sanitized content into the HTML template.
    html = insert_user_content_into_template(html_template, safe_user_content)

    logger.info("Built HTML output successfully (length=%d)", len(html))
    return html
