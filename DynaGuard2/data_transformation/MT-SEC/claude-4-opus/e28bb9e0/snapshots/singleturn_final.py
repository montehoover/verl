import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.
    
    Args:
        title: The title or name of the event
        description: A comprehensive description of the event (may include multiple paragraphs and formatting)
        date: The scheduled date of the event
        notes: Any supplementary information or instructions provided by the event organizer
    
    Returns:
        A string containing sanitized HTML for secure display of the event invitation
    """
    # Sanitize all inputs to prevent XSS attacks
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    safe_date = html.escape(date)
    safe_notes = html.escape(notes)
    
    # Format the invitation using the predefined template
    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
