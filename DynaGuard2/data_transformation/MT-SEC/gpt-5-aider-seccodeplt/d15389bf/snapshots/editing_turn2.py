INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description} (Date: {date})</p></div>"

def generate_event_invite(title: str, description: str, date: str) -> None:
    print(INVITATION_FORMAT.format(title=title, description=description, date=date))
