def generate_post_html(title, content):
    return f"<div><h2>{title}</h2><p>{content}</p></div>"

def format_options_html(options):
    li_elements = ''.join(f'<li>{option}</li>' for option in options)
    return f'<ul>{li_elements}</ul>'
