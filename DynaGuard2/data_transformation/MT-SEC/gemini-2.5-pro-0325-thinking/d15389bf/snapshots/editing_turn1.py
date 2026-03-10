INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p></div>"

def generate_event_invite(title: str, description: str):
    """
    Creates an HTML format for event invitations and prints it.

    Args:
        title: The name of the event.
        description: An overview of the event.
    """
    html_invite = INVITATION_FORMAT.format(title=title, description=description)
    print(html_invite)
