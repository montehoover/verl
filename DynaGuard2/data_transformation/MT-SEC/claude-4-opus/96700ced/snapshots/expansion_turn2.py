def create_command(template):
    """
    Formats a base command template by substituting placeholder values.
    
    Args:
        template (str): A shell command template with placeholders
        
    Returns:
        str: A complete command string after performing substitutions
    """
    # Example implementation assuming placeholders like {placeholder_name}
    # and that placeholder values are available in a global context
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    # Create the command by replacing placeholders
    command = template
    for placeholder in placeholders:
        # Assuming placeholder values are available globally
        # You would replace this with your actual global lookup mechanism
        if placeholder in globals():
            command = command.replace(f'{{{placeholder}}}', str(globals()[placeholder]))
    
    return command


def insert_name_into_command(command_template, name_string):
    """
    Safely incorporates a name string into a command template.
    
    Args:
        command_template (str): A shell command template containing name placeholder
        name_string (str): The name to insert into the template
        
    Returns:
        str: The complete shell command string with the name included
    """
    import shlex
    
    # Escape the name string for safe shell usage
    safe_name = shlex.quote(name_string)
    
    # Replace common name placeholders
    command = command_template.replace('{name}', safe_name)
    command = command.replace('{{name}}', safe_name)
    command = command.replace('$NAME', safe_name)
    command = command.replace('${NAME}', safe_name)
    
    return command
