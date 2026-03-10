FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title, post_body, options_list):
    options_html = "<ul>\n"
    for option in options_list:
        options_html += f"  <li>{option}</li>\n"
    options_html += "</ul>"
    
    return FORUM_POST_FORMAT.format(
        title=title,
        post_body=post_body,
        options_list=options_html
    )
