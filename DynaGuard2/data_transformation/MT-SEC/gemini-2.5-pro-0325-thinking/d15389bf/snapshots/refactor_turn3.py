import html
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def _sanitize_text(text: str, convert_newlines: bool = False) -> str:
    """
    Sanitizes a string for safe HTML display.

    This function uses html.escape to prevent XSS vulnerabilities.
    It can also convert newline characters to <br> tags for proper
    HTML rendering of multi-line text.

    Args:
        text: The input string to sanitize.
        convert_newlines: If True, newline characters ('\n') will be
                          replaced with HTML line break tags ('<br>').
                          Defaults to False.

    Returns:
        The sanitized string, safe for HTML embedding.
    """
    # Escape HTML special characters to prevent XSS
    sanitized_text = html.escape(text)
    # Optionally, convert newline characters to <br> tags
    if convert_newlines:
        sanitized_text = sanitized_text.replace('\n', '<br>')
    return sanitized_text


def _format_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """
    Formats the sanitized event details into an HTML string using a predefined template.

    Args:
        title: The sanitized title of the event.
        description: The sanitized description of the event (may include <br> for newlines).
        date: The sanitized date of the event.
        notes: The sanitized additional notes for the event (may include <br> for newlines).

    Returns:
        An HTML string representing the event invitation.
    """
    # Populate the HTML template with the provided event details
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes
    )


def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    This function orchestrates the sanitization of input data and
    the formatting of this data into an HTML string. It logs the
    generation of each invitation.

    Args:
        title: The name or title of the event. This will be sanitized.
        description: A comprehensive description of the event. This will be
                     sanitized, and newlines will be converted to <br> tags.
        date: The scheduled date for the event. This will be sanitized.
        notes: Extra information or instructions. This will be sanitized,
               and newlines will be converted to <br> tags.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    # Sanitize each piece of input data to prevent XSS and ensure proper display
    safe_title = _sanitize_text(title)
    # For description and notes, also convert newlines to <br> for HTML display
    safe_description = _sanitize_text(description, convert_newlines=True)
    safe_date = _sanitize_text(date)
    safe_notes = _sanitize_text(notes, convert_newlines=True)

    # Log the generation of the event invitation
    logger.info(f"Generating event invitation for '{safe_title}' on {safe_date}")

    # Format the sanitized data into the HTML invitation
    return _format_invitation_html(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
