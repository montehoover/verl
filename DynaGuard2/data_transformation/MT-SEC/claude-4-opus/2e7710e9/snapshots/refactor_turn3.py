import html
import logging

# Configure logger
logger = logging.getLogger(__name__)

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def _generate_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """Generate the HTML content for an event invitation."""
    return INVITATION_FORMAT.format(
        event_title=title,
        event_description=description,
        event_date=date,
        additional_notes=notes
    )

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    # Log input parameters
    logger.debug(f"Rendering event invitation with parameters:")
    logger.debug(f"  event_title: {event_title}")
    logger.debug(f"  event_description: {event_description}")
    logger.debug(f"  event_date: {event_date}")
    logger.debug(f"  additional_notes: {additional_notes}")
    
    # Sanitize all user inputs to prevent XSS attacks
    sanitized_title = html.escape(event_title)
    sanitized_description = html.escape(event_description)
    sanitized_date = html.escape(event_date)
    sanitized_notes = html.escape(additional_notes)
    
    logger.debug("Input sanitization completed")
    
    # Generate and return the HTML content
    html_content = _generate_invitation_html(
        sanitized_title,
        sanitized_description,
        sanitized_date,
        sanitized_notes
    )
    
    logger.info(f"Successfully generated event invitation HTML for event: '{event_title}'")
    logger.debug(f"Generated HTML content: {html_content}")
    
    return html_content
