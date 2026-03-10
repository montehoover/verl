import re
import html

def replace_placeholders(template, value_range):
    # Find all placeholders in the template
    placeholders = re.findall(r'\{\{[^}]+\}\}', template)
    
    if not placeholders:
        return template
    
    # Calculate evenly distributed values
    num_placeholders = len(placeholders)
    if num_placeholders == 1:
        values = [(value_range[0] + value_range[1]) / 2]
    else:
        step = (value_range[1] - value_range[0]) / (num_placeholders - 1)
        values = [value_range[0] + i * step for i in range(num_placeholders)]
    
    # Check if any values are decimals
    for value in values:
        if not value.is_integer():
            raise ValueError("Placeholder replacement results in decimal values")
    
    # Replace placeholders with values
    result = template
    for placeholder, value in zip(placeholders, values):
        result = result.replace(placeholder, str(int(value)), 1)
    
    # Check if all placeholders were replaced
    remaining_placeholders = re.findall(r'\{\{[^}]+\}\}', result)
    if remaining_placeholders:
        raise ValueError("Not all placeholders were replaced")
    
    return result

def create_html_template(html_template, user_values):
    # Find all placeholders in the template
    placeholders = re.findall(r'\{\{([^}]+)\}\}', html_template)
    
    # Replace each placeholder with its corresponding value
    result = html_template
    for placeholder in placeholders:
        if placeholder in user_values:
            # Escape the user value for safe HTML output
            safe_value = html.escape(str(user_values[placeholder]))
            result = result.replace('{{' + placeholder + '}}', safe_value)
    
    return result
