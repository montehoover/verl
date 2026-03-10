import re

def construct_command(base_command, parameters):
    """
    Constructs a full command string from a base command and a list of parameters.
    
    Args:
        base_command (str): The base command string
        parameters (list): List of parameters to append to the command
        
    Returns:
        str: The full command with all parameters concatenated
        
    Raises:
        ValueError: If the command contains invalid characters or is malformed
    """
    # Validate base_command
    if not base_command or not isinstance(base_command, str):
        raise ValueError("Base command must be a non-empty string")
    
    if not base_command.strip():
        raise ValueError("Base command cannot be only whitespace")
    
    # Check for invalid characters in base command
    # Allow alphanumeric, spaces, hyphens, underscores, dots, and slashes
    if not re.match(r'^[a-zA-Z0-9\s\-_./]+$', base_command):
        raise ValueError(f"Base command contains invalid characters: {base_command}")
    
    # Validate parameters
    if not isinstance(parameters, list):
        raise ValueError("Parameters must be a list")
    
    # Start with the base command
    full_command = base_command.strip()
    
    # Add each parameter separated by spaces
    for i, param in enumerate(parameters):
        param_str = str(param).strip()
        
        if not param_str:
            raise ValueError(f"Parameter at index {i} is empty or only whitespace")
        
        # Check for dangerous characters that could lead to command injection
        # Disallow semicolons, pipes, backticks, dollar signs, and other shell metacharacters
        dangerous_chars = r'[;&|`$<>\\]'
        if re.search(dangerous_chars, param_str):
            raise ValueError(f"Parameter '{param_str}' contains potentially dangerous characters")
        
        full_command += " " + param_str
    
    # Final validation of the complete command
    if len(full_command) > 1000:  # Arbitrary limit to prevent extremely long commands
        raise ValueError("Constructed command is too long (max 1000 characters)")
    
    return full_command


# Example usage
if __name__ == "__main__":
    # Test the function with valid inputs
    try:
        base = "git"
        params = ["commit", "-m", "Initial commit"]
        result = construct_command(base, params)
        print(result)  # Output: git commit -m Initial commit
    except ValueError as e:
        print(f"Error: {e}")
    
    # Another example
    try:
        base2 = "python"
        params2 = ["script.py", "--verbose", "--output", "result.txt"]
        result2 = construct_command(base2, params2)
        print(result2)  # Output: python script.py --verbose --output result.txt
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with invalid inputs
    try:
        base3 = "rm -rf /"
        params3 = ["; echo", "malicious"]
        result3 = construct_command(base3, params3)
        print(result3)
    except ValueError as e:
        print(f"Error: {e}")  # Should raise error for semicolon
    
    # Test with empty base command
    try:
        base4 = ""
        params4 = ["test"]
        result4 = construct_command(base4, params4)
        print(result4)
    except ValueError as e:
        print(f"Error: {e}")  # Should raise error for empty base command
