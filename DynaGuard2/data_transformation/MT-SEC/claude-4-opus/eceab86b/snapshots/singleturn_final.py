import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def create_invitation_for_event(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.
    
    Args:
        event_name: str, the name or title of the event
        description_of_event: str, a comprehensive description of the event (may contain multiple lines and formatting)
        date_of_event: str, the scheduled date for the event
        notes_for_event: str, extra information or instructions provided by the user
    
    Returns:
        A string containing sanitized HTML for safely displaying the event invitation
    """
    # Sanitize all input strings to prevent XSS attacks
    safe_event_name = html.escape(event_name)
    safe_description = html.escape(description_of_event)
    safe_date = html.escape(date_of_event)
    safe_notes = html.escape(notes_for_event)
    
    # Format the invitation using the predefined template
    invitation_html = INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description,
        date_of_event=safe_date,
        notes_for_event=safe_notes
    )
    
    return invitation_html
