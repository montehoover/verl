import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def create_event_summary(title: str, date: str, description: str) -> str:
    """
    Formats event details into an HTML string.

    Args:
        title: The title of the event.
        date: The date of the event.
        description: The description of the event.

    Returns:
        An HTML string with the event details in separate div blocks.
    """
    html_output = f"""
<div>{title}</div>
<div>{date}</div>
<div>{description}</div>
"""
    return html_output.strip()


def sanitize_notes(notes: str) -> str:
    """
    Sanitizes a string to make it HTML-safe.

    Args:
        notes: The string to sanitize.

    Returns:
        An HTML-safe version of the input string.
    """
    return html.escape(notes)


def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a complete HTML event invitation.

    Args:
        title: The title of the event.
        description: The description of the event.
        date: The date of the event.
        notes: Additional notes for the event.

    Returns:
        A securely structured HTML string for the event invitation.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    safe_date = html.escape(date)
    safe_notes = sanitize_notes(notes)  # Uses the existing sanitize_notes function

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
