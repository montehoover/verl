import html

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"


def generate_event_html(event_title: str, event_date: str) -> str:
    """
    Generates an HTML representation of an event.

    Args:
        event_title: The title of the event.
        event_date: The date of the event.

    Returns:
        An HTML string representing the event.
    """
    return f"<div><h2>{event_title}</h2><p><strong>Date:</strong> {event_date}</p></div>"


def format_notes_html(notes: str) -> str:
    """
    Formats a string of notes to be HTML-safe.

    Args:
        notes: The string of notes to format.

    Returns:
        An HTML-safe string of notes.
    """
    return html.escape(notes)


def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Generates a full HTML view of an event invitation.

    Args:
        event_title: The title of the event.
        event_description: The description of the event.
        event_date: The date of the event.
        additional_notes: Additional notes for the event.

    Returns:
        A complete HTML string for the event invitation.
    """
    safe_additional_notes = format_notes_html(additional_notes)
    return INVITATION_FORMAT.format(
        event_title=event_title,
        event_description=event_description,
        event_date=event_date,
        additional_notes=safe_additional_notes
    )
