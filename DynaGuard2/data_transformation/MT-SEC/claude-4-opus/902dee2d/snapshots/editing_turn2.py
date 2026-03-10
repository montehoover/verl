def forum_post_with_list(headline, content, items):
    post = f"# {headline}\n\n{content}\n\n"
    for item in items:
        post += f"- {item}\n"
    return post
