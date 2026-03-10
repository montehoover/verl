import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Displays a user-generated post in a public forum.

    Args:
        username: The name of the user who created the post.
        post_title: The title of the post.
        post_body: The main content of the post.
        tags: A list of tags added by the user.

    Returns:
        A sanitized post in HTML format containing user information.
    """
    sanitized_username = html.escape(username)
    sanitized_post_title = html.escape(post_title)
    sanitized_post_body = html.escape(post_body)

    sanitized_tags = [f"<span>{html.escape(tag)}</span>" for tag in tags]
    tags_html = " ".join(sanitized_tags)

    return USER_POST_FORMAT.format(
        post_title=sanitized_post_title,
        username=sanitized_username,
        post_body=sanitized_post_body,
        tags_html=tags_html
    )

if __name__ == '__main__':
    # Example Usage:
    example_input = {
        "username": "JohnDoe",
        "post_title": "My First Post",
        "post_body": "Hello, this is my first post!",
        "tags": [
            "Introduction",
            "FirstPost"
        ]
    }
    output_html = render_user_post(
        example_input["username"],
        example_input["post_title"],
        example_input["post_body"],
        example_input["tags"]
    )
    print(output_html)

    example_input_xss = {
        "username": "<script>alert('xss')</script>User",
        "post_title": "XSS <img src=x onerror=alert('title')>",
        "post_body": "Body with <iframe src='javascript:alert(\"body\")'></iframe>",
        "tags": [
            "SafeTag",
            "<script>alert('tag')</script>"
        ]
    }
    output_html_xss = render_user_post(
        example_input_xss["username"],
        example_input_xss["post_title"],
        example_input_xss["post_body"],
        example_input_xss["tags"]
    )
    print("\n--- XSS Attempt Example ---")
    print(output_html_xss)
