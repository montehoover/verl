import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def create_invitation_for_event(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    Args:
        event_name: The name or title of the event.
        description_of_event: A comprehensive description of the event.
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    safe_event_name = html.escape(event_name)
    safe_description_of_event = html.escape(description_of_event)
    safe_date_of_event = html.escape(date_of_event)
    safe_notes_for_event = html.escape(notes_for_event)

    return INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description_of_event,
        date_of_event=safe_date_of_event,
        notes_for_event=safe_notes_for_event
    )
