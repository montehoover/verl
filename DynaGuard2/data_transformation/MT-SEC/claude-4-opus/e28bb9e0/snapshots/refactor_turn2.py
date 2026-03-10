import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def sanitize_text(text: str) -> str:
    """Sanitize text to prevent XSS attacks."""
    return html.escape(text)

def format_invitation(title: str, description: str, date: str, notes: str) -> str:
    """Format the invitation using the predefined template."""
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes
    )

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    # Sanitize all inputs
    safe_title = sanitize_text(title)
    safe_description = sanitize_text(description)
    safe_date = sanitize_text(date)
    safe_notes = sanitize_text(notes)
    
    # Format and return the invitation
    return format_invitation(safe_title, safe_description, safe_date, safe_notes)
