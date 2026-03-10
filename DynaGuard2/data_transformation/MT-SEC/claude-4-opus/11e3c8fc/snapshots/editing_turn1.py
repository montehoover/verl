def basic_post_json(author_id, post_heading):
    return {
        "title": post_heading,
        "content": author_id
    }
