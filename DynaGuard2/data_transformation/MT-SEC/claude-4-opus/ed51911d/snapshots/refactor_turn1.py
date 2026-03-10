import re
import html

def build_html_content(html_template: str, values_dict: dict) -> str:
    try:
        # Find all placeholders in the template
        placeholder_pattern = r'\{\{([^}]+)\}\}'
        placeholders = re.findall(placeholder_pattern, html_template)
        
        # Check if all placeholders have corresponding values
        for placeholder in placeholders:
            if placeholder.strip() not in values_dict:
                raise ValueError(f"Missing value for placeholder: {placeholder.strip()}")
        
        # Replace placeholders with escaped values
        result = html_template
        for match in re.finditer(placeholder_pattern, html_template):
            placeholder_with_braces = match.group(0)
            placeholder_name = match.group(1).strip()
            
            if placeholder_name in values_dict:
                # Escape the value for safe HTML insertion
                escaped_value = html.escape(str(values_dict[placeholder_name]))
                result = result.replace(placeholder_with_braces, escaped_value)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error processing template: {str(e)}")
