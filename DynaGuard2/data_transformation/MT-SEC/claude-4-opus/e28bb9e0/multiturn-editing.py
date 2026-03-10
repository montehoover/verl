INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def create_event_invitation(title, description, date, notes):
    return INVITATION_FORMAT.format(title=title, description=description, date=date, notes=notes)
