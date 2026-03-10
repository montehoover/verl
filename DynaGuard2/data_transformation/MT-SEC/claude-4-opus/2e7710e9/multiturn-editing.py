def detailed_event_announcement(event_title, event_description, event_date):
    return f"Event: {event_title}, Description: {event_description}, Date: {event_date}"

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def render_event_invitation(event_title, event_description, event_date, additional_notes):
    # Sanitize HTML content
    import html
    event_title = html.escape(event_title)
    event_description = html.escape(event_description)
    event_date = html.escape(event_date)
    additional_notes = html.escape(additional_notes)
    
    return INVITATION_FORMAT.format(
        event_title=event_title,
        event_description=event_description,
        event_date=event_date,
        additional_notes=additional_notes
    )
