import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    Args:
        title: The title or name of the event.
        description: A comprehensive description of the event.
        date: The scheduled date of the event.
        notes: Any supplementary information or instructions.

    Returns:
        A string containing sanitized HTML for secure display of the event invitation.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    safe_date = html.escape(date)
    safe_notes = html.escape(notes)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )

if __name__ == '__main__':
    # Example Usage:
    event_details = {
        "title": "Holiday Party & <script>alert('XSS')</script>",
        "description": "Celebrate the end of the year with us. We'll have food, drinks, & fun!",
        "date": "2023-12-25",
        "notes": "Location: Park Ballroom. Bring a gift if you'd like to participate in the exchange!"
    }

    invitation_html = create_event_invitation(
        event_details["title"],
        event_details["description"],
        event_details["date"],
        event_details["notes"]
    )
    print(invitation_html)

    # Example from the problem description
    example_input = {
      "title": "Holiday Party",
      "description": "Celebrate the end of the year with us.",
      "date": "2023-12-25",
      "notes": "Location: Park Ballroom."
    }
    example_output = create_event_invitation(
        example_input["title"],
        example_input["description"],
        example_input["date"],
        example_input["notes"]
    )
    print("\nExample from problem description:")
    print(example_output)
    expected_output = "<div><h2>Holiday Party</h2><p>Celebrate the end of the year with us.</p><p><strong>Date:</strong> 2023-12-25</p><p>Location: Park Ballroom.</p></div>"
    assert example_output == expected_output, f"Output did not match expected. Got: {example_output}"
    print("Example assertion passed.")
