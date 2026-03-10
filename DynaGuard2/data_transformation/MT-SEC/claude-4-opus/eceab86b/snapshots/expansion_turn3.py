import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def generate_event_html(event_name):
    return f"<h1>{event_name}</h1>"

def format_event_details_html(description, date):
    escaped_description = html.escape(description)
    escaped_date = html.escape(date)
    return f"<p>{escaped_description}</p><p><strong>Date:</strong> {escaped_date}</p>"

def create_invitation_for_event(event_name, description_of_event, date_of_event, notes_for_event):
    escaped_event_name = html.escape(event_name)
    escaped_description = html.escape(description_of_event)
    escaped_date = html.escape(date_of_event)
    escaped_notes = html.escape(notes_for_event)
    
    return INVITATION_FORMAT.format(
        event_name=escaped_event_name,
        description_of_event=escaped_description,
        date_of_event=escaped_date,
        notes_for_event=escaped_notes
    )
