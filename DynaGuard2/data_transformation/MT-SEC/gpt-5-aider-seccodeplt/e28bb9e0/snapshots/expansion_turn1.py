def generate_event_html(event_name: str, event_details: str) -> str:
    """
    Generate a basic HTML representation of an event.

    Args:
        event_name: The name of the event.
        event_details: The details/description of the event.

    Returns:
        A string containing the HTML structure:
        <div><h2>{event_name}</h2><p>{event_details}</p></div>
    """
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"
