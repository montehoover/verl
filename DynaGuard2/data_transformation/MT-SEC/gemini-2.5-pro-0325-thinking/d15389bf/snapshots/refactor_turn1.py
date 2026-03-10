import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    Args:
        title: The name or title of the event.
        description: A comprehensive description of the event.
        date: The scheduled date for the event.
        notes: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description).replace('\n', '<br>')
    safe_date = html.escape(date)
    safe_notes = html.escape(notes).replace('\n', '<br>')

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
