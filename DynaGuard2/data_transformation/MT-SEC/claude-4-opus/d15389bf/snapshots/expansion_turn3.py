INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def create_event_summary(title, date, description):
    """
    Create an HTML string with event details in separate div blocks.
    
    Args:
        title (str): The event title
        date (str): The event date
        description (str): The event description
    
    Returns:
        str: HTML string with formatted event details
    """
    html = f"""<div class="event-summary">
    <div class="event-title">{title}</div>
    <div class="event-date">{date}</div>
    <div class="event-description">{description}</div>
</div>"""
    return html


def sanitize_notes(notes):
    """
    Sanitize user input to make it HTML-safe.
    
    Args:
        notes (str): The raw notes input
    
    Returns:
        str: HTML-safe version of the notes
    """
    # Replace HTML special characters with their entities
    sanitized = notes.replace("&", "&amp;")
    sanitized = sanitized.replace("<", "&lt;")
    sanitized = sanitized.replace(">", "&gt;")
    sanitized = sanitized.replace('"', "&quot;")
    sanitized = sanitized.replace("'", "&#x27;")
    return sanitized


def generate_event_invite(title, description, date, notes):
    """
    Generate a complete HTML event invitation with sanitized content.
    
    Args:
        title (str): The event title
        description (str): The event description
        date (str): The event date
        notes (str): Additional notes for the event
    
    Returns:
        str: Complete HTML invitation string
    """
    # Sanitize all user inputs to prevent HTML injection
    safe_title = sanitize_notes(title)
    safe_description = sanitize_notes(description)
    safe_date = sanitize_notes(date)
    safe_notes = sanitize_notes(notes)
    
    # Generate the invitation using the predefined format
    invitation = INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
    
    return invitation
