import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def create_invitation_for_event(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> str:
    """
    Creates an HTML invitation for an event with sanitized inputs.

    Args:
        event_name: The name of the event.
        description_of_event: The description of the event, may include multiple lines.
        date_of_event: The date of the event.
        notes_for_event: Additional notes for the event.

    Returns:
        A string containing sanitized HTML suitable for an event invitation.
    """
    safe_event_name = html.escape(event_name)
    # Replace newlines with <br> for HTML display after escaping
    safe_description_of_event = html.escape(description_of_event).replace('\n', '<br>')
    safe_date_of_event = html.escape(date_of_event)
    safe_notes_for_event = html.escape(notes_for_event).replace('\n', '<br>')

    return INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description_of_event,
        date_of_event=safe_date_of_event,
        notes_for_event=safe_notes_for_event
    )
