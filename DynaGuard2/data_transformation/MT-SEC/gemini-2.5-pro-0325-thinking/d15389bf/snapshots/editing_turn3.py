INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Creates a sanitized HTML format for event invitations and returns it.

    Args:
        title: The name of the event.
        description: An overview of the event.
        date: The date of the event.
        notes: Additional information or notes for the event.

    Returns:
        A sanitized HTML string representing the event invitation.
    """
    html_invite = INVITATION_FORMAT.format(title=title, description=description, date=date, notes=notes)
    return html_invite
