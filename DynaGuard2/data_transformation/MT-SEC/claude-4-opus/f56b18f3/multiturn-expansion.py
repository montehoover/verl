BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_image_caption(image_alt, caption):
    return f'<div><img alt="{image_alt}" /><p>{caption}</p></div>'

def compose_bio_section(name, bio):
    # Escape HTML special characters to prevent layout breaks
    name = name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    bio = bio.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    return f'<section><h2>{name}</h2><p>{bio}</p></section>'

def create_blog_post(heading, text):
    # Escape HTML special characters
    heading = heading.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    return BLOG_FORMAT.format(heading=heading, text=text)
