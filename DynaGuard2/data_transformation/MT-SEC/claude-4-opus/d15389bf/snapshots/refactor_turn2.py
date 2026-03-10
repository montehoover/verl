import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def sanitize_input(text: str) -> str:
    """Sanitize a single input string to prevent XSS attacks."""
    return html.escape(text)

def sanitize_event_data(title: str, description: str, date: str, notes: str) -> dict:
    """Sanitize all event data and return as a dictionary."""
    return {
        'title': sanitize_input(title),
        'description': sanitize_input(description),
        'date': sanitize_input(date),
        'notes': sanitize_input(notes)
    }

def format_invitation(event_data: dict) -> str:
    """Format sanitized event data into HTML using the invitation template."""
    return INVITATION_FORMAT.format(**event_data)

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    # Sanitize all event data
    sanitized_data = sanitize_event_data(title, description, date, notes)
    
    # Format the sanitized data into HTML
    return format_invitation(sanitized_data)
