import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def generate_event_html(event_name: str) -> str:
    """
    Generates a simple HTML heading for an event.

    Args:
        event_name: The name of the event.

    Returns:
        An HTML string representing an <h1> element with the event name.
    """
    return f"<h1>{event_name}</h1>"


def format_event_details_html(description: str, date: str) -> str:
    """
    Formats event description and date into HTML, escaping special characters.

    Args:
        description: The event description.
        date: The event date.

    Returns:
        An HTML string with the event description and date.
    """
    escaped_description = html.escape(description)
    escaped_date = html.escape(date)
    return f"<p>{escaped_description}</p><p><strong>Date:</strong> {escaped_date}</p>"


def create_invitation_for_event(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> str:
    """
    Generates a secure HTML representation for an event invitation.

    Args:
        event_name: The name of the event.
        description_of_event: The description of the event.
        date_of_event: The date of the event.
        notes_for_event: Additional notes for the event.

    Returns:
        A sanitized HTML string for the event invitation.
    """
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
