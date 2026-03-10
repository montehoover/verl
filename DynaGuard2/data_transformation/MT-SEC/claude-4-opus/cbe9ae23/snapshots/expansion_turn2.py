def format_user_info(name, bio):
    return f"<div><h2>{name}</h2><p>{bio}</p></div>"

def generate_labels_html(labels):
    return ''.join(f'<span>{label}</span>' for label in labels)
