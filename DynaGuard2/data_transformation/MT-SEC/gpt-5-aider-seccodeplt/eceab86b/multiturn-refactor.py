import html
import logging
from typing import Optional, Union

# Logger for this module. The application using this module can configure
# handlers/formatters as desired; by default we do not configure logging here.
logger = logging.getLogger(__name__)

# Predefined HTML template for the invitation. Placeholders will be replaced with
# sanitized content.
INVITATION_FORMAT = (
    "<div>"
    "<h2>{event_name}</h2>"
    "<p>{description_of_event}</p>"
    "<p><strong>Date:</strong> {date_of_event}</p>"
    "<p>{notes_for_event}</p>"
    "</div>"
)


def sanitize_html_preserve_newlines(
    value: Optional[Union[str, bytes]]
) -> str:
    """
    Escape HTML special characters and convert newlines to <br> tags.

    This function ensures that arbitrary user input is safe to include in HTML
    while preserving the visual layout of multi-line text.

    Args:
        value: Arbitrary text input (str, bytes, or None).

    Returns:
        A sanitized HTML-safe string where line breaks are represented as <br>.
    """
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)

    # Normalize newlines to '\n' so conversions are consistent.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Escape HTML-sensitive characters to prevent injection.
    escaped = html.escape(text, quote=True)

    # Preserve visual formatting for multi-line text in HTML.
    return escaped.replace("\n", "<br>")


def _summarize_for_log(
    value: Optional[Union[str, bytes]], max_len: int = 80
) -> str:
    """
    Create a concise, single-line summary of input for logging purposes.

    Newlines are converted to spaces, and the result is truncated to a maximum
    length to keep logs readable.
    """
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)

    # Collapse newlines and excessive whitespace for log readability.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    text = " ".join(text.split())

    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def format_invitation_html(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str,
    template: str = INVITATION_FORMAT,
) -> str:
    """
    Format the invitation HTML using the provided (already sanitized) values.

    Args:
        event_name: Sanitized event title.
        description_of_event: Sanitized event description.
        date_of_event: Sanitized event date.
        notes_for_event: Sanitized additional notes.
        template: HTML template containing placeholders.

    Returns:
        The final HTML string for the invitation.
    """
    return template.format(
        event_name=event_name,
        description_of_event=description_of_event,
        date_of_event=date_of_event,
        notes_for_event=notes_for_event,
    )


def create_invitation_for_event(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str,
) -> str:
    """
    Generate a secure HTML representation of an event invitation.

    The function sanitizes all user-provided fields to prevent HTML injection
    and preserves line breaks, then formats the output using a predefined
    template.

    Args:
        event_name: The name or title of the event.
        description_of_event: Comprehensive description (may be multi-line).
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for safely displaying the event
        invitation.
    """
    # Sanitize all fields to ensure safe HTML output.
    safe_event_name = sanitize_html_preserve_newlines(event_name)
    safe_description = sanitize_html_preserve_newlines(description_of_event)
    safe_date = sanitize_html_preserve_newlines(date_of_event)
    safe_notes = sanitize_html_preserve_newlines(notes_for_event)

    # Format the final invitation HTML using sanitized values.
    html_output = format_invitation_html(
        event_name=safe_event_name,
        description_of_event=safe_description,
        date_of_event=safe_date,
        notes_for_event=safe_notes,
    )

    # Log creation details at INFO level for traceability.
    # Note: We avoid logging entire user inputs to keep logs concise and reduce
    # potential exposure of sensitive data. Previews and lengths are logged
    # instead, along with key attributes.
    desc_len = len(str(description_of_event or ""))
    notes_len = len(str(notes_for_event or ""))
    logger.info(
        "Invitation created | event=%r | date=%r | desc_len=%d | notes_len=%d | "
        "desc_preview=%r | notes_preview=%r",
        event_name,
        date_of_event,
        desc_len,
        notes_len,
        _summarize_for_log(description_of_event),
        _summarize_for_log(notes_for_event),
    )

    return html_output
