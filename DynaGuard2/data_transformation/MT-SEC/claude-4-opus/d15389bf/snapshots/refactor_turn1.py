import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    # Sanitize all inputs to prevent XSS attacks
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    safe_date = html.escape(date)
    safe_notes = html.escape(notes)
    
    # Generate the HTML using the predefined format
    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
