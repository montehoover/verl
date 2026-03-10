import re
import html

def extract_placeholders(template_string):
    """
    Identifies placeholders in an HTML template.

    Args:
        template_string: A string representing an HTML template.

    Returns:
        A list of placeholder names found within double curly braces.
    """
    # Regex to find {{placeholder_name}}
    # It captures the content inside the double curly braces
    placeholder_pattern = r"\{\{(.*?)\}\}"
    placeholders = re.findall(placeholder_pattern, template_string)
    return placeholders

def replace_placeholders(template_string, placeholder_values):
    """
    Replaces placeholders in an HTML template with user-provided values.

    Args:
        template_string: A string representing an HTML template.
        placeholder_values: A dictionary mapping placeholder names to their values.

    Returns:
        The HTML string with placeholders replaced by their corresponding values,
        with values HTML-escaped.
    """
    modified_template = template_string
    for placeholder_name, value in placeholder_values.items():
        # Construct the placeholder pattern, e.g., {{name}}
        placeholder_tag = f"{{{{{placeholder_name}}}}}"
        # Escape the value to prevent HTML injection
        escaped_value = html.escape(str(value))
        modified_template = modified_template.replace(placeholder_tag, escaped_value)
    return modified_template

def render_html_template(html_template, input_map):
    """
    Processes an HTML template, replacing placeholders with user-provided values.

    Args:
        html_template: A string with the HTML template.
        input_map: A dictionary mapping placeholder names to their values.

    Returns:
        The final HTML string with placeholders replaced.

    Raises:
        ValueError: If a placeholder in the template is not found in input_map,
                    or if any other processing error occurs.
    """
    try:
        # Find all unique placeholders in the template
        template_placeholders = extract_placeholders(html_template)
        
        # Check if all placeholders in the template have corresponding values in input_map
        for placeholder_name in template_placeholders:
            if placeholder_name not in input_map:
                raise ValueError(f"Missing value for placeholder: {placeholder_name}")

        # Replace placeholders with their escaped values
        # We can reuse the logic from replace_placeholders, but ensure we only act on
        # placeholders that were actually in the template and confirmed to be in input_map.
        # A direct replacement loop is also clear here.
        rendered_html = html_template
        for placeholder_name in template_placeholders:
            value = input_map[placeholder_name]
            escaped_value = html.escape(str(value))
            placeholder_tag = f"{{{{{placeholder_name}}}}}"
            # Ensure all occurrences of a placeholder are replaced
            rendered_html = rendered_html.replace(placeholder_tag, escaped_value)
            
        return rendered_html
    except ValueError as ve:
        # Re-raise ValueError specifically for missing placeholders or other ValueErrors
        raise ve
    except Exception as e:
        # Catch any other unexpected errors during processing and wrap them in a ValueError
        raise ValueError(f"Error processing HTML template: {e}")


if __name__ == '__main__':
    # Example usage:
    sample_template = "<html><body><h1>Hello, {{name}}!</h1><p>Welcome to {{place}}.</p></body></html>"
    found_placeholders = extract_placeholders(sample_template)
    print(f"Found placeholders: {found_placeholders}")

    sample_template_2 = "No placeholders here."
    found_placeholders_2 = extract_placeholders(sample_template_2)
    print(f"Found placeholders in second template: {found_placeholders_2}")

    sample_template_3 = "<html><body><p>{{user_id}} logged in from {{ip_address}}.</p></body></html>"
    found_placeholders_3 = extract_placeholders(sample_template_3)
    print(f"Found placeholders in third template: {found_placeholders_3}")

    # Example usage for replace_placeholders:
    data_for_template_1 = {"name": "Alice", "place": "Wonderland"}
    rendered_template_1 = replace_placeholders(sample_template, data_for_template_1)
    print(f"\nRendered template 1: {rendered_template_1}")

    data_for_template_3 = {"user_id": 123, "ip_address": "192.168.1.1"}
    rendered_template_3 = replace_placeholders(sample_template_3, data_for_template_3)
    print(f"Rendered template 3: {rendered_template_3}")

    # Example with HTML characters in data
    data_with_html = {"name": "<script>alert('XSS')</script>", "place": "<b>Bold Place</b>"}
    rendered_template_html_chars = replace_placeholders(sample_template, data_with_html)
    print(f"Rendered template with HTML chars (escaped): {rendered_template_html_chars}")

    # Example with a placeholder not in the data dictionary (should remain unchanged)
    template_with_extra_placeholder = "Hello {{name}}, welcome to {{city}} in {{country}}."
    data_for_partial_replacement = {"name": "Bob", "country": "USA"}
    rendered_partial_template = replace_placeholders(template_with_extra_placeholder, data_for_partial_replacement)
    print(f"Rendered partial template: {rendered_partial_template}")

    # Example usage for render_html_template:
    print("\n--- render_html_template examples ---")
    template_rt = "<html><body><h1>Hello, {{name}}!</h1><p>Your age is {{age}}.</p></body></html>"
    data_rt_success = {"name": "Charlie", "age": 30}
    try:
        rendered_rt_success = render_html_template(template_rt, data_rt_success)
        print(f"Rendered template (success): {rendered_rt_success}")
    except ValueError as e:
        print(f"Error (success case, unexpected): {e}")

    data_rt_missing_placeholder = {"name": "David"} # 'age' is missing
    try:
        rendered_rt_fail = render_html_template(template_rt, data_rt_missing_placeholder)
        print(f"Rendered template (failure case, should not print): {rendered_rt_fail}")
    except ValueError as e:
        print(f"Error (missing placeholder, expected): {e}")

    template_rt_no_placeholders = "No placeholders here."
    data_rt_no_placeholders = {}
    try:
        rendered_rt_no_placeholders = render_html_template(template_rt_no_placeholders, data_rt_no_placeholders)
        print(f"Rendered template (no placeholders): {rendered_rt_no_placeholders}")
    except ValueError as e:
        print(f"Error (no placeholders case, unexpected): {e}")

    template_rt_with_html_injection = "<p>Hello {{name}}</p>"
    data_rt_with_html_injection = {"name": "<script>alert('bad');</script>"}
    try:
        rendered_rt_html_injection = render_html_template(template_rt_with_html_injection, data_rt_with_html_injection)
        print(f"Rendered template (HTML injection escaped): {rendered_rt_html_injection}")
    except ValueError as e:
        print(f"Error (HTML injection case, unexpected): {e}")
    
    # Example with duplicate placeholders in template
    template_duplicate_placeholders = "<p>{{item}} and {{item}}</p>"
    data_duplicate_placeholders = {"item": "Apple"}
    try:
        rendered_duplicate = render_html_template(template_duplicate_placeholders, data_duplicate_placeholders)
        print(f"Rendered template (duplicate placeholders): {rendered_duplicate}")
    except ValueError as e:
        print(f"Error (duplicate placeholders, unexpected): {e}")
