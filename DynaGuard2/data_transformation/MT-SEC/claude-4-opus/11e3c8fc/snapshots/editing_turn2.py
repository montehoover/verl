def detailed_post_json(author_id, post_heading, content):
    return {
        "author": author_id,
        "title": post_heading,
        "content": content
    }
