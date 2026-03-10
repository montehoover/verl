import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def create_invitation_for_event(event_name, description_of_event, date_of_event, notes_for_event):
    return INVITATION_FORMAT.format(
        event_name=html.escape(event_name),
        description_of_event=html.escape(description_of_event),
        date_of_event=html.escape(date_of_event),
        notes_for_event=html.escape(notes_for_event)
    )
