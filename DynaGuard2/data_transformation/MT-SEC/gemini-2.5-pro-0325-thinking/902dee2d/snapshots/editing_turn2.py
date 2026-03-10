from typing import List

def forum_post_with_list(headline: str, content: str, items: List[str]) -> str:
    """
    Creates a markdown structure for a forum post, including a list of items.

    Args:
        headline: The headline of the post.
        content: The content of the post.
        items: A list of strings to be formatted as an unordered list.

    Returns:
        A markdown formatted string.
    """
    post_string = f"# {headline}\n\n{content}"
    if items:
        post_string += "\n\n"
        for item in items:
            post_string += f"- {item}\n"
    return post_string.strip()
