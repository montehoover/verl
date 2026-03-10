INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def generate_event_html(event_name, event_details):
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"

def format_event_details(description, date, notes):
    return f"{description} <strong>{date}</strong><p>{notes}</p>"

def create_event_invitation(title, description, date, notes):
    return INVITATION_FORMAT.format(title=title, description=description, date=date, notes=notes)
