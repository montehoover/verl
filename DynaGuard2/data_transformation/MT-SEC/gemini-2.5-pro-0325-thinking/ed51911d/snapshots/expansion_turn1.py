import re

def parse_placeholders(html_string):
    """
    Parses an HTML string to find placeholders formatted as {{...}}.

    Args:
        html_string (str): The HTML string to parse.

    Returns:
        list: A list of placeholder names (the content within {{...}}),
              with leading/trailing whitespace stripped from each name.
              Returns an empty list if no placeholders are found.
    """
    # Regex to find {{placeholder_name}}
    # The (.*?) part is a non-greedy match for any characters inside the braces.
    placeholder_regex = r"\{\{(.*?)\}\}"
    placeholders_found = re.findall(placeholder_regex, html_string)
    # Strip whitespace from each found placeholder name
    return [p.strip() for p in placeholders_found]

# Example usage (can be removed or commented out if not needed as part of the library code)
if __name__ == '__main__':
    test_cases = [
        ("<h1>Hello {{name}}!</h1><p>Welcome to {{city}}.</p>", ['name', 'city']),
        ("<p>Your score is {{ score }}. Next level: {{ level.next }}.</p>", ['score', 'level.next']),
        ("No placeholders here.", []),
        ("{{first}} {{second_placeholder}} {{third.dot.notation}} {{fourth with spaces}}", ['first', 'second_placeholder', 'third.dot.notation', 'fourth with spaces']),
        ("Outer {{ data_key }} with {{ obj.property }} and maybe {{ func(arg) }}.", ['data_key', 'obj.property', 'func(arg)']),
        # This case demonstrates how non-greedy findall handles content that itself contains braces,
        # which is a common interpretation of "nested" in this context.
        # It finds 'a {{nested}} example' as one placeholder name.
        ("This is {{a {{nested}} example}} and {{another}}.", ['a {{nested}} example', 'another']),
        # Similarly for these:
        ("{{a {{b}} c}}", ['a {{b}} c']),
        ("{{ {{a}} }}", ['{{a}}']), # Content is ' {{a}} ', stripped to '{{a}}'
        ("{{a}} {{b {{c}} d}} {{e}}", ['a', 'b {{c}} d', 'e']),
        ("{{}}", ['']), # Empty placeholder name
        ("{{ }}", ['']), # Empty placeholder name after strip
        ("text {{ var1 }} then {{var2}} end", ['var1', 'var2'])
    ]

    for i, (html, expected) in enumerate(test_cases):
        result = parse_placeholders(html)
        print(f"Test Case {i+1}:")
        print(f"  Input: \"{html}\"")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        assert result == expected, f"Test Case {i+1} Failed!"
        print(f"  Status: Passed")
        print("-" * 20)

    print("All tests passed.")
