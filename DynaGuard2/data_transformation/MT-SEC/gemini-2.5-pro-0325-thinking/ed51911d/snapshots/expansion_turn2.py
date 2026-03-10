import re
import html

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

def replace_placeholders(template_string, values, default_value=""):
    """
    Replaces placeholders in an HTML template string with values from a dictionary.

    Args:
        template_string (str): The HTML template string with {{...}} placeholders.
        values (dict): A dictionary where keys are placeholder names and values are their replacements.
        default_value (str, optional): The value to use if a placeholder is not found in the values dictionary.
                                     Defaults to an empty string.

    Returns:
        str: The HTML string with placeholders replaced. Values are HTML-escaped.
    """
    if not isinstance(template_string, str):
        raise TypeError("template_string must be a string")
    if not isinstance(values, dict):
        raise TypeError("values must be a dictionary")
    if not isinstance(default_value, str):
        # Coerce to string or raise error, for now coerce for flexibility
        default_value = str(default_value)

    placeholder_regex = r"\{\{(.*?)\}\}"

    def replacer(match):
        placeholder_name = match.group(1).strip()
        # Get value from dict, or use default_value if not found
        value_to_insert = values.get(placeholder_name, default_value)
        # Ensure the value is a string before escaping
        return html.escape(str(value_to_insert))

    return re.sub(placeholder_regex, replacer, template_string)

# Example usage (can be removed or commented out if not needed as part of the library code)
if __name__ == '__main__':
    # Test cases for parse_placeholders
    print("--- Testing parse_placeholders ---")
    parse_test_cases = [
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

    for i, (html_input, expected) in enumerate(parse_test_cases):
        result = parse_placeholders(html_input)
        print(f"Parse Test Case {i+1}:")
        print(f"  Input: \"{html_input}\"")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        assert result == expected, f"Parse Test Case {i+1} Failed!"
        print(f"  Status: Passed")
        print("-" * 20)
    print("All parse_placeholders tests passed.\n")

    # Test cases for replace_placeholders
    print("--- Testing replace_placeholders ---")
    replace_test_cases = [
        ("<h1>Hello {{name}}!</h1>", {"name": "World"}, "DEFAULT", "<h1>Hello World!</h1>"),
        ("<p>Welcome to {{city}}.</p>", {"city": "New York"}, "DEFAULT", "<p>Welcome to New York.</p>"),
        ("User: {{user}}, Age: {{age}}", {"user": "Alice"}, "N/A", "User: Alice, Age: N/A"),
        ("Data: {{data}}", {}, "Missing", "Data: Missing"),
        ("{{tag}}", {"tag": "<script>alert('XSS')</script>"}, "", "&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;"),
        ("No placeholders here.", {"value": "test"}, "DEFAULT", "No placeholders here."),
        ("{{ a }} and {{ b }}", {"a": "1", "b": "2"}, "", "1 and 2"),
        ("{{ a }} and {{b.c}}", {"a": "1", "b.c": "nested_val"}, "", "1 and nested_val"),
        ("Test with {{ empty_val }}", {"empty_val": ""}, "DEF", "Test with "),
        ("Test with {{ num_val }}", {"num_val": 123}, "DEF", "Test with 123"),
        ("Test with {{ bool_val }}", {"bool_val": True}, "DEF", "Test with True"),
        ("{{key1}} {{key2}} {{key1}}", {"key1": "A", "key2": "B"}, "X", "A B A"),
        ("Price: ${{price}}", {"price": "10.99"}, "N/A", "Price: $10.99"), # Check no double escaping of $
        ("{{ unescaped }}", {"unescaped": "R&D"}, "", "R&amp;D"),
    ]

    for i, (template, values, default, expected_output) in enumerate(replace_test_cases):
        result = replace_placeholders(template, values, default)
        print(f"Replace Test Case {i+1}:")
        print(f"  Template: \"{template}\"")
        print(f"  Values: {values}")
        print(f"  Default: \"{default}\"")
        print(f"  Expected: \"{expected_output}\"")
        print(f"  Got: \"{result}\"")
        assert result == expected_output, f"Replace Test Case {i+1} Failed!"
        print(f"  Status: Passed")
        print("-" * 20)

    # Test type errors for replace_placeholders
    print("--- Testing replace_placeholders type errors ---")
    try:
        replace_placeholders(123, {}, "")
    except TypeError as e:
        print(f"Caught expected TypeError for template_string: {e}")
        assert "template_string must be a string" in str(e)
    else:
        assert False, "TypeError not raised for invalid template_string type"

    try:
        replace_placeholders("test", "not a dict", "")
    except TypeError as e:
        print(f"Caught expected TypeError for values: {e}")
        assert "values must be a dictionary" in str(e)
    else:
        assert False, "TypeError not raised for invalid values type"
    
    # Test default_value coercion (should not raise error, but coerce)
    result_coerced_default = replace_placeholders("{{test}}", {}, 123)
    expected_coerced_default = "123"
    print(f"Test Coerced Default: Input '{{test}}', {{}}, 123")
    print(f"  Expected: \"{expected_coerced_default}\"")
    print(f"  Got: \"{result_coerced_default}\"")
    assert result_coerced_default == expected_coerced_default, "Default value coercion failed"
    print(f"  Status: Passed")
    print("-" * 20)


    print("All replace_placeholders tests (including type checks) passed.")
