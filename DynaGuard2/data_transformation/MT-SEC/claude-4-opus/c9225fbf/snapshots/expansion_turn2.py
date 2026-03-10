def generate_user_content_html(author_name, content):
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"

def format_tags(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)
