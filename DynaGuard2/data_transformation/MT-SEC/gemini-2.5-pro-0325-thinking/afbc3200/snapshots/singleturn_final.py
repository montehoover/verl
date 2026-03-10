FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Creates a structured HTML representation of a forum post.

    Args:
        post_title: str, the title or main topic of the forum post.
        post_body: str, the main body or content of the forum post.
        bullet_points: list(str), a collection of user-provided discussion points.

    Returns:
        A string containing the fully formatted HTML representation of the forum post.
    """
    options_html = "<ul>"
    for point in bullet_points:
        options_html += f"<li>{point}</li>"
    options_html += "</ul>"

    return FORUM_POST_FORMAT.format(header=post_title, message=post_body, options=options_html)

if __name__ == '__main__':
    # Example Usage:
    example_input = {
        "post_title": "Best Pizza Toppings?",
        "post_body": "Let us discuss our favorite pizza toppings.",
        "bullet_points": [
            "Pepperoni",
            "Mushrooms",
            "Extra Cheese"
        ]
    }
    
    output_html = build_forum_entry(
        example_input["post_title"],
        example_input["post_body"],
        example_input["bullet_points"]
    )
    
    expected_output = "<h2>Best Pizza Toppings?</h2><p>Let us discuss our favorite pizza toppings.</p><ul><li>Pepperoni</li><li>Mushrooms</li><li>Extra Cheese</li></ul>"
    
    print("Generated HTML:")
    print(output_html)
    print("\nExpected HTML:")
    print(expected_output)
    print(f"\nMatches expected: {output_html == expected_output}")

    # Example with no bullet points
    example_input_no_bullets = {
        "post_title": "Welcome!",
        "post_body": "Welcome to the forum.",
        "bullet_points": []
    }

    output_html_no_bullets = build_forum_entry(
        example_input_no_bullets["post_title"],
        example_input_no_bullets["post_body"],
        example_input_no_bullets["bullet_points"]
    )

    expected_output_no_bullets = "<h2>Welcome!</h2><p>Welcome to the forum.</p><ul></ul>"
    
    print("\nGenerated HTML (no bullets):")
    print(output_html_no_bullets)
    print("\nExpected HTML (no bullets):")
    print(expected_output_no_bullets)
    print(f"\nMatches expected (no bullets): {output_html_no_bullets == expected_output_no_bullets}")
