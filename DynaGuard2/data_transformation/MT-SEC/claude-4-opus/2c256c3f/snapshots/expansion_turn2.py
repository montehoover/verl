def create_profile_html(writer_name, bio):
    return f"<div><h2>{writer_name}</h2><p>{bio}</p></div>"

def wrap_tags_in_html(tags):
    return ' '.join(f'<span>{tag}</span>' for tag in tags)
