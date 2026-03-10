from html import escape as _html_escape

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def normalize_newlines(value: str) -> str:
    """
    Normalize various newline representations to '\n'.
    Ensures input is a string and coerces None to empty string.
    """
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def escape_html(value: str) -> str:
    """
    Escape special HTML characters to their HTML-safe equivalents.
    """
    return _html_escape(value, quote=True)


def newlines_to_br(value: str) -> str:
    """
    Convert newline characters to <br> tags for HTML display.
    """
    return value.replace("\n", "<br>")


def escape_braces_for_format(value: str) -> str:
    """
    Escape braces to prevent str.format from interpreting them as placeholders.
    """
    return value.replace("{", "{{").replace("}", "}}")


def sanitize_plain_text_for_html(value: str) -> str:
    """
    Sanitize text for safe HTML display without converting newlines.
    - Normalizes newlines
    - Escapes HTML special characters
    - Escapes braces for str.format safety
    """
    v = normalize_newlines(value)
    v = escape_html(v)
    v = escape_braces_for_format(v)
    return v


def sanitize_text_with_newlines_for_html(value: str) -> str:
    """
    Sanitize text for safe HTML display with newline conversion.
    - Normalizes newlines
    - Escapes HTML special characters
    - Converts newlines to <br>
    - Escapes braces for str.format safety
    """
    v = normalize_newlines(value)
    v = escape_html(v)
    v = newlines_to_br(v)
    v = escape_braces_for_format(v)
    return v


def format_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """
    Format sanitized fields into the invitation HTML template.
    """
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes,
    )


def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a sanitized HTML string for an event invitation.

    Args:
        title: The title or name of the event.
        description: A comprehensive description of the event (may include multiple paragraphs).
        date: The scheduled date of the event.
        notes: Any supplementary information or instructions provided by the organizer.

    Returns:
        A string containing sanitized HTML for secure display of the event invitation.
    """
    safe_title = sanitize_plain_text_for_html(title)
    safe_description = sanitize_text_with_newlines_for_html(description)
    safe_date = sanitize_plain_text_for_html(date)
    safe_notes = sanitize_text_with_newlines_for_html(notes)

    return format_invitation_html(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
