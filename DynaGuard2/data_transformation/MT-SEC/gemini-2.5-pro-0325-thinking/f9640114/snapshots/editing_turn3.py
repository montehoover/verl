from typing import List

FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title: str, post_body: str, options_list: List[str]) -> str:
    """
    Builds an HTML string for a forum post.

    Args:
        title: The title of the post.
        post_body: The main content of the post.
        options_list: A list of strings to be presented as an unordered list.

    Returns:
        A sanitized HTML string representing the forum post.
    """
    options_html = "<ul>\n"
    for option in options_list:
        options_html += f"  <li>{option}</li>\n"
    options_html += "</ul>"

    return FORUM_POST_FORMAT.format(title=title, post_body=post_body, options_list=options_html)
