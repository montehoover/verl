import html

def generate_event_html(event_name):
    return f"<h1>{event_name}</h1>"

def format_event_details_html(description, date):
    escaped_description = html.escape(description)
    escaped_date = html.escape(date)
    return f"<p>{escaped_description}</p><p><strong>Date:</strong> {escaped_date}</p>"
