INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p></div>"

def generate_event_invite(title: str, description: str) -> None:
    print(INVITATION_FORMAT.format(title=title, description=description))
