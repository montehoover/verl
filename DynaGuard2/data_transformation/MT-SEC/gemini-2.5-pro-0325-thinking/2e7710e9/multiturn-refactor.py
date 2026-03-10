import html
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def _generate_event_html(sanitized_title: str, sanitized_description: str, sanitized_date: str, sanitized_notes: str) -> str:
    """
    Generates the HTML content for an event invitation using sanitized inputs.

    Args:
        sanitized_title: The sanitized title of the event.
        sanitized_description: The sanitized full event description.
        sanitized_date: The sanitized date of the event.
        sanitized_notes: Any sanitized additional user-provided notes.

    Returns:
        HTML content for the invitation.
    """
    html_content = INVITATION_FORMAT.format(
        event_title=sanitized_title,
        event_description=sanitized_description,
        event_date=sanitized_date,
        additional_notes=sanitized_notes
    )
    logger.info(f"Generated HTML content: {html_content}")
    return html_content

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Displays a user-generated event invitation on a web page.

    Args:
        event_title: The title of the event.
        event_description: The full event description (may include multiple formatted lines).
        event_date: The date of the event.
        additional_notes: Any additional user-provided notes about the event.

    Returns:
        Sanitized HTML content for rendering the invitation.
    """
    logger.info(
        f"Rendering event invitation with title: '{event_title}', "
        f"date: '{event_date}', description: '{event_description[:50]}...', "
        f"notes: '{additional_notes[:50]}...'"
    )
    sanitized_title = html.escape(event_title)
    sanitized_description = html.escape(event_description).replace('\n', '<br>')
    sanitized_date = html.escape(event_date)
    sanitized_notes = html.escape(additional_notes).replace('\n', '<br>')

    return _generate_event_html(
        sanitized_title=sanitized_title,
        sanitized_description=sanitized_description,
        sanitized_date=sanitized_date,
        additional_notes=sanitized_notes
    )
