import html

# Predefined HTML template string
INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a sanitized HTML string for an event invitation.

    Args:
        title: The title of the event.
        description: The description of the event.
        date: The date of the event.
        notes: Additional notes for the event.

    Returns:
        A sanitized HTML formatted string for the event invitation.
    """
    # Sanitize inputs to prevent XSS
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    safe_date = html.escape(date)
    safe_notes = html.escape(notes)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
