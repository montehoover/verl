import re
import html

def extract_placeholders(template_string):
    """
    Identifies placeholders in an HTML template.

    Args:
        template_string: A string representing an HTML template.

    Returns:
        A list of placeholder names found within {{...}}.
        Returns an empty list if no placeholders are found.
    """
    # Regex to find {{placeholder_name}}
    # It captures the content inside the double curly braces
    placeholder_pattern = r"\{\{(.*?)\}\}"
    placeholders = re.findall(placeholder_pattern, template_string)
    return placeholders

def replace_placeholders(template_string, values):
    """
    Replaces placeholders in an HTML template with user-provided values.

    Args:
        template_string: A string representing an HTML template.
        values: A dictionary where keys are placeholder names and
                values are the strings to replace them with.

    Returns:
        The HTML content with placeholders replaced by corresponding,
        HTML-escaped values.
    """
    output_string = template_string
    for placeholder_name, value in values.items():
        # Escape the value to prevent XSS
        escaped_value = html.escape(str(value))
        # Construct the placeholder pattern, e.g., {{name}}
        pattern_to_replace = r"\{\{" + re.escape(placeholder_name) + r"\}\}"
        output_string = re.sub(pattern_to_replace, escaped_value, output_string)
    return output_string

def create_html_template(html_template, user_values):
    """
    Processes an HTML template by replacing placeholders with user-provided values.

    Args:
        html_template: A string with the HTML template.
        user_values: A dictionary mapping placeholder names to their values.

    Returns:
        A string with the generated HTML content.

    Raises:
        ValueError: If any placeholder in the template is not found in user_values,
                    or if any other error occurs during processing.
    """
    try:
        placeholders_in_template = extract_placeholders(html_template)
        
        for placeholder in placeholders_in_template:
            if placeholder not in user_values:
                raise ValueError(f"Missing value for placeholder: {placeholder}")
        
        # All required placeholders are present, proceed with replacement
        processed_html = html_template
        for placeholder_name, value in user_values.items():
            # Escape the value to prevent XSS
            escaped_value = html.escape(str(value))
            # Construct the placeholder pattern, e.g., {{name}}
            pattern_to_replace = r"\{\{" + re.escape(placeholder_name) + r"\}\}"
            # Replace all occurrences of this placeholder
            processed_html = re.sub(pattern_to_replace, escaped_value, processed_html)
            
        return processed_html
    except ValueError as ve:
        # Re-raise ValueError specifically for missing placeholders
        raise ve
    except Exception as e:
        # Catch any other potential errors during processing
        raise ValueError(f"An error occurred during HTML template processing: {e}")

if __name__ == '__main__':
    # Example Usage
    html_template1 = "<html><body><h1>Hello, {{name}}!</h1><p>Welcome to {{place}}.</p></body></html>"
    html_template2 = "<html><body><p>This is a static page.</p></body></html>"
    html_template3 = ""
    html_template4_xss = "<html><body><p>{{comment}}</p></body></html>"

    print(f"Placeholders in template 1: {extract_placeholders(html_template1)}")
    print(f"Placeholders in template 2: {extract_placeholders(html_template2)}")
    print(f"Placeholders in template 3: {extract_placeholders(html_template3)}")

    # Example for replace_placeholders
    values1 = {"name": "Alice", "place": "Wonderland"}
    print(f"\nReplaced template 1: {replace_placeholders(html_template1, values1)}")

    values2 = {"name": "Bob"} # Not all placeholders are filled
    print(f"Replaced template 1 (partial): {replace_placeholders(html_template1, values2)}")

    values_xss = {"comment": "<script>alert('XSS')</script>"}
    print(f"Replaced template 4 (XSS attempt): {replace_placeholders(html_template4_xss, values_xss)}")

    values_empty = {}
    print(f"Replaced template 1 (empty values): {replace_placeholders(html_template1, values_empty)}")

    # Example for create_html_template
    print("\n--- create_html_template examples ---")
    template_a = "<div>{{greeting}}, {{user}}. Your score is {{score}}!</div>"
    values_a_complete = {"greeting": "Hi", "user": "Tester", "score": 100}
    values_a_incomplete = {"greeting": "Hello", "user": "Tester"} # Missing 'score'
    values_a_extra = {"greeting": "Hey", "user": "Admin", "score": 99, "extra_field": "ignored"}


    try:
        print(f"Generated HTML (complete): {create_html_template(template_a, values_a_complete)}")
    except ValueError as e:
        print(f"Error (complete): {e}")

    try:
        print(f"Generated HTML (incomplete): {create_html_template(template_a, values_a_incomplete)}")
    except ValueError as e:
        print(f"Error (incomplete): {e}")

    try:
        print(f"Generated HTML (extra values): {create_html_template(template_a, values_a_extra)}")
    except ValueError as e:
        print(f"Error (extra values): {e}")

    template_b_no_placeholders = "<p>No placeholders here.</p>"
    values_b = {"name": "NoOne"}
    try:
        print(f"Generated HTML (no placeholders in template): {create_html_template(template_b_no_placeholders, values_b)}")
    except ValueError as e:
        print(f"Error (no placeholders in template): {e}")

    template_c_script = "<p>{{content}}</p>"
    values_c_script = {"content": "<script>doEvil()</script>"}
    try:
        print(f"Generated HTML (script content): {create_html_template(template_c_script, values_c_script)}")
    except ValueError as e:
        print(f"Error (script content): {e}")
