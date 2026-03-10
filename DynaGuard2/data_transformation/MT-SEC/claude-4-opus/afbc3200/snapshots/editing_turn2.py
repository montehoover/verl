def basic_forum_post(post_title, post_body, discussion_points):
    points_str = ''.join([f"<item>{point}</item>" for point in discussion_points])
    return f"<title>{post_title}</title><content>{post_body}</content><discussion>{points_str}</discussion>"
