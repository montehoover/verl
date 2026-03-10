import html

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def generate_event_html(event_title, event_date):
    return f"<div><h2>{event_title}</h2><p><strong>Date:</strong> {event_date}</p></div>"

def format_notes_html(notes):
    return html.escape(notes)

def render_event_invitation(event_title, event_description, event_date, additional_notes):
    return INVITATION_FORMAT.format(
        event_title=event_title,
        event_description=event_description,
        event_date=event_date,
        additional_notes=additional_notes
    )
