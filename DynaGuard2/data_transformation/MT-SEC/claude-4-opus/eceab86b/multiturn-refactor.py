import html
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent XSS attacks.
    
    Args:
        text: The raw text input to sanitize
        
    Returns:
        HTML-escaped version of the input text
    """
    return html.escape(text)


def format_invitation(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str
) -> str:
    """
    Format the invitation using the predefined HTML template.
    
    Args:
        event_name: The sanitized event name
        description_of_event: The sanitized event description
        date_of_event: The sanitized event date
        notes_for_event: The sanitized additional notes
        
    Returns:
        Formatted HTML string for the invitation
    """
    return INVITATION_FORMAT.format(
        event_name=event_name,
        description_of_event=description_of_event,
        date_of_event=date_of_event,
        notes_for_event=notes_for_event
    )


def create_invitation_for_event(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str
) -> str:
    """
    Generate a secure HTML representation of an event invitation.
    
    This function takes event details and creates a sanitized HTML invitation
    that is safe to display in a web context. All user inputs are escaped
    to prevent XSS attacks.
    
    Args:
        event_name: The name or title of the event
        description_of_event: A comprehensive description of the event
                             (may contain multiple lines and formatting)
        date_of_event: The scheduled date for the event
        notes_for_event: Extra information or instructions provided by the user
        
    Returns:
        A string containing sanitized HTML for safely displaying the event invitation
    """
    # Log the start of invitation creation
    logger.info(
        f"Creating invitation for event: '{event_name}' scheduled on {date_of_event}"
    )
    
    # Sanitize all user inputs to prevent XSS attacks
    safe_event_name = sanitize_input(event_name)
    safe_description = sanitize_input(description_of_event)
    safe_date = sanitize_input(date_of_event)
    safe_notes = sanitize_input(notes_for_event)
    
    # Format and return the invitation using sanitized inputs
    invitation = format_invitation(
        safe_event_name,
        safe_description,
        safe_date,
        safe_notes
    )
    
    # Log successful invitation creation with details
    logger.info(
        f"Successfully generated invitation | "
        f"Event: '{event_name}' | "
        f"Date: {date_of_event} | "
        f"Description length: {len(description_of_event)} chars | "
        f"Has notes: {'Yes' if notes_for_event.strip() else 'No'}"
    )
    
    return invitation
