import html
from typing import Dict

# Predefined HTML template for the event invitation.
# Placeholders {event_name}, {description_of_event}, {date_of_event},
# and {notes_for_event} will be replaced with sanitized event data.
INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def _sanitize_event_details(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> Dict[str, str]:
    """
    Sanitizes event details to prevent XSS vulnerabilities.

    Args:
        event_name: The name or title of the event.
        description_of_event: A comprehensive description of the event.
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A dictionary containing sanitized event details.
    """
    return {
        "event_name": html.escape(event_name),
        "description_of_event": html.escape(description_of_event),
        "date_of_event": html.escape(date_of_event),
        "notes_for_event": html.escape(notes_for_event),
    }


def _format_invitation_html(sanitized_details: Dict[str, str]) -> str:
    """
    Formats the sanitized event details into an HTML string.

    Args:
        sanitized_details: A dictionary containing sanitized event details.
                           Expected keys: "event_name", "description_of_event",
                           "date_of_event", "notes_for_event".

    Returns:
        A string containing the formatted HTML invitation.
    """
    return INVITATION_FORMAT.format(
        event_name=sanitized_details["event_name"],
        description_of_event=sanitized_details["description_of_event"],
        date_of_event=sanitized_details["date_of_event"],
        notes_for_event=sanitized_details["notes_for_event"]
    )


def create_invitation_for_event(event_name: str, description_of_event: str, date_of_event: str, notes_for_event: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    This function first sanitizes the input event details to prevent
    cross-site scripting (XSS) attacks. Then, it formats these
    sanitized details into an HTML string using a predefined template.

    Args:
        event_name: The name or title of the event.
        description_of_event: A comprehensive description of the event.
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    # Sanitize all input strings to prevent XSS.
    sanitized_details = _sanitize_event_details(
        event_name, description_of_event, date_of_event, notes_for_event
    )

    # Format the sanitized details into the HTML invitation.
    invitation_html = _format_invitation_html(sanitized_details)

    return invitation_html
