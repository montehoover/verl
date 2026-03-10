import html

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
    # Sanitize all user inputs to prevent XSS attacks
    sanitized_title = html.escape(event_title)
    sanitized_description = html.escape(event_description)
    sanitized_date = html.escape(event_date)
    sanitized_notes = html.escape(additional_notes)
    
    # Generate and return the HTML content
    return _generate_invitation_html(
        sanitized_title,
        sanitized_description,
        sanitized_date,
        sanitized_notes
    )
