import html

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Renders an HTML event invitation.

    Args:
        event_title: The title of the event.
        event_description: The description of the event.
        event_date: The date of the event.
        additional_notes: Additional notes for the event.

    Returns:
        A sanitized HTML string for the event invitation.
    """
    return INVITATION_FORMAT.format(
        event_title=html.escape(event_title),
        event_description=html.escape(event_description),
        event_date=html.escape(event_date),
        additional_notes=html.escape(additional_notes)
    )
