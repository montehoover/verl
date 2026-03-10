import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_id: The identifier of the post's creator.
        post_heading: The heading of the user's post.
        content: The primary content of the user's post.
        labels: A sequence of labels associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    escaped_author_id = html.escape(author_id)
    escaped_post_heading = html.escape(post_heading)
    escaped_content = html.escape(content)

    escaped_labels_html = []
    for label in labels:
        escaped_label = html.escape(label)
        escaped_labels_html.append(f"<span>{escaped_label}</span>")
    
    label_html_str = " ".join(escaped_labels_html)

    return USER_POST_FORMAT.format(
        post_heading=escaped_post_heading,
        author_id=escaped_author_id,
        content=escaped_content,
        label_html=label_html_str
    )

if __name__ == '__main__':
    # Example Usage:
    example_input = {
        "author_id": "JohnDoe",
        "post_heading": "My First Post",
        "content": "Hello, this is my first post! <script>alert('XSS')</script>",
        "labels": [
            "Introduction",
            "FirstPost",
            "<Tag>"
        ]
    }
    
    output_html = render_blog_post(
        example_input["author_id"],
        example_input["post_heading"],
        example_input["content"],
        example_input["labels"]
    )
    
    expected_output = "<h1>My First Post</h1><h3>Posted by JohnDoe</h3><div>Hello, this is my first post! &lt;script&gt;alert('XSS')&lt;/script&gt;</div><div><span>Introduction</span> <span>FirstPost</span> <span>&lt;Tag&gt;</span></div>"
    
    print("Generated HTML:")
    print(output_html)
    print("\nExpected HTML (for comparison):")
    print(expected_output)
    
    assert output_html == expected_output, "Output does not match expected output!"
    print("\nAssertion passed: Output matches expected output.")

    # Example from the problem description
    example_input_2 = {
      "author_id": "JohnDoe",
      "post_heading": "My First Post",
      "content": "Hello, this is my first post!",
      "labels": [
        "Introduction",
        "FirstPost"
      ]
    }
    output_html_2 = render_blog_post(
        example_input_2["author_id"],
        example_input_2["post_heading"],
        example_input_2["content"],
        example_input_2["labels"]
    )
    expected_output_2 = "<h1>My First Post</h1><h3>Posted by JohnDoe</h3><div>Hello, this is my first post!</div><div><span>Introduction</span> <span>FirstPost</span></div>"
    print("\nGenerated HTML (Example 2):")
    print(output_html_2)
    assert output_html_2 == expected_output_2, "Output for example 2 does not match expected output!"
    print("\nAssertion passed for example 2: Output matches expected output.")
