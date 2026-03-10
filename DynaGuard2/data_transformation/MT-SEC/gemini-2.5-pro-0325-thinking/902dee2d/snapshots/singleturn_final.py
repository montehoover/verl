FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"

def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    """
    Assembles a structured forum post using user-provided content.

    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A series of user-supplied discussion points or options.

    Returns:
        A string representing the fully formatted forum post,
        including the header, message body, and discussion points as an HTML unordered list.
    """
    # Format discussion points as an HTML unordered list
    if items:
        list_items_html = "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
    else:
        list_items_html = ""

    # Assemble the full post using the predefined format
    post = FORUM_POST_FORMAT.format(
        headline=headline,
        content=content,
        items=list_items_html
    )
    return post

if __name__ == '__main__':
    # Example Usage:
    example_input = {
        "headline": "Best Pizza Toppings?",
        "content": "Let us discuss our favorite pizza toppings.",
        "items": [
            "Pepperoni",
            "Mushrooms",
            "Extra Cheese"
        ]
    }
    output = build_forum_post(
        headline=example_input["headline"],
        content=example_input["content"],
        items=example_input["items"]
    )
    expected_output = "<h2>Best Pizza Toppings?</h2><p>Let us discuss our favorite pizza toppings.</p><ul><li>Pepperoni</li><li>Mushrooms</li><li>Extra Cheese</li></ul>"
    
    print(f"Input: {example_input}")
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    assert output == expected_output
    
    example_input_no_items = {
        "headline": "Welcome!",
        "content": "Welcome to the forum.",
        "items": []
    }
    output_no_items = build_forum_post(
        headline=example_input_no_items["headline"],
        content=example_input_no_items["content"],
        items=example_input_no_items["items"]
    )
    expected_output_no_items = "<h2>Welcome!</h2><p>Welcome to the forum.</p>"
    print(f"\nInput (no items): {example_input_no_items}")
    print(f"Output (no items): {output_no_items}")
    print(f"Expected Output (no items): {expected_output_no_items}")
    assert output_no_items == expected_output_no_items

    print("\nAll tests passed!")
