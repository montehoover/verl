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
