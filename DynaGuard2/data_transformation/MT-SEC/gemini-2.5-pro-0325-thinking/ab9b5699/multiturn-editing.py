import re
import html
from typing import Tuple, Dict

def replace_placeholders(text_template: str, value_range: Tuple[float, float]) -> str:
    """
    Replaces placeholders in a text template with evenly distributed values from a given range.

    Args:
        text_template: The string template with placeholders in {{...}} format.
        value_range: A tuple (start_value, end_value) defining the range of values.

    Returns:
        The text_template with placeholders replaced by calculated values.

    Raises:
        ValueError: If a placeholder replacement results in a decimal value,
                    or if not all placeholders are replaced.
    """
    # Find all occurrences of placeholders like {{anything}}
    # Using re.finditer to get match objects if we needed placeholder names,
    # but re.findall is sufficient if we only need to count them and replace them in order.
    placeholders = re.findall(r"\{\{.*?\}\}", text_template)
    num_placeholders = len(placeholders)

    if num_placeholders == 0:
        return text_template

    start_value, end_value = float(value_range[0]), float(value_range[1])

    generated_values = []
    if num_placeholders == 1:
        # If there's only one placeholder, it takes the start_value.
        # Alternative interpretations could be the end_value or the midpoint.
        # This interpretation aligns with generating N points where the first is start_value.
        generated_values.append(start_value)
    else:
        # Generate N evenly distributed values from start_value to end_value (inclusive)
        step = (end_value - start_value) / (num_placeholders - 1)
        for i in range(num_placeholders):
            generated_values.append(start_value + i * step)
        # Ensure the last value is precisely end_value to counteract potential float inaccuracies
        generated_values[-1] = end_value
    
    # Iterator for replacement values
    values_iter = iter(generated_values)

    def replacer(match_obj):
        # The content of the placeholder (e.g., the '...' in {{...}}) is not used.
        # Each placeholder is replaced sequentially with the next generated value.
        try:
            value = next(values_iter)
            # Check if the value is a decimal (float with a fractional part)
            if isinstance(value, float) and value % 1 != 0:
                raise ValueError(f"Placeholder replacement resulted in a decimal value: {value}")
            # Format the value as a string.
            return str(int(value)) if isinstance(value, float) and value % 1 == 0 else str(value)
        except StopIteration:
            # This should not be reached if num_placeholders matches generated_values count.
            # If it does, it means re.sub found more placeholders than initially counted,
            # or generated_values was exhausted prematurely.
            # Return original placeholder; the post-replacement check will catch this.
            return match_obj.group(0)

    # Substitute all placeholders using the replacer function
    modified_text = re.sub(r"\{\{.*?\}\}", replacer, text_template)

    # Check if any placeholders remain in the modified text
    if re.search(r"\{\{.*?\}\}", modified_text):
        raise ValueError("Not all placeholders were replaced. Check template or replacement logic.")
    
    return modified_text

def create_html_template(html_template: str, user_values: Dict[str, any]) -> str:
    """
    Replaces placeholders in an HTML template with user-provided values.

    Placeholders are in the format {{placeholder_name}}.
    User-provided values are HTML-escaped before insertion.

    Args:
        html_template: The string HTML template with placeholders.
        user_values: A dictionary mapping placeholder names (str) to their values.

    Returns:
        The HTML template with placeholders replaced by their corresponding, escaped values.

    Raises:
        ValueError: If a placeholder in the template is not found in user_values.
    """
    def replacer(match_obj):
        # Extract the placeholder name, stripping any leading/trailing whitespace
        # from within the curly braces. e.g., {{ my_key }} -> "my_key"
        placeholder_name = match_obj.group(1).strip()
        
        if placeholder_name in user_values:
            value = user_values[placeholder_name]
            # Escape the value to prevent HTML injection and convert to string
            return html.escape(str(value))
        else:
            # If a placeholder in the template is not in user_values, raise an error.
            raise ValueError(f"Placeholder '{{{{{placeholder_name}}}}}' not found in user_values.")

    # Substitute all placeholders using the replacer function.
    # The regex r"\{\{(.*?)\}\}" captures the content inside {{...}}.
    return re.sub(r"\{\{(.*?)\}\}", replacer, html_template)
