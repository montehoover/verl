from typing import List

def basic_forum_post(post_title: str, post_body: str, discussion_points: List[str]) -> str:
    """
    Creates a simple XML representation of a forum post.

    Args:
        post_title: The title of the post.
        post_body: The body content of the post.
        discussion_points: A list of strings representing discussion points.

    Returns:
        A string with the title in a <title> tag, the body in a <content> tag,
        and discussion points in <item> tags within a <discussion_points> tag.
    """
    points_xml = "\n".join([f"    <item>{point}</item>" for point in discussion_points])
    return f"<post>\n  <title>{post_title}</title>\n  <content>{post_body}</content>\n  <discussion_points>\n{points_xml}\n  </discussion_points>\n</post>"
