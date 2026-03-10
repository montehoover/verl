def generate_event_html(event_name, event_details):
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"

def format_event_details(description, date, notes):
    return f"{description} <strong>{date}</strong><p>{notes}</p>"
