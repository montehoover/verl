BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading, text):
    return BLOG_FORMAT.format(heading=heading, text=text)
