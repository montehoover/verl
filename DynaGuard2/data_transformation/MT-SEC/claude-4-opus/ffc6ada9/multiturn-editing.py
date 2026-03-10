import subprocess
import shlex
import re

def execute_command(base_command, parameters):
    """
    Constructs and executes a shell command from a base command and parameters.
    
    Args:
        base_command (str): The base command string
        parameters (list): List of parameters to append to the command
        
    Returns:
        str: The output from the executed command
        
    Raises:
        ValueError: If the command contains invalid characters, is malformed, or cannot be executed
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
    command_parts = [base_command.strip()]
    
    # Add each parameter
    for i, param in enumerate(parameters):
        param_str = str(param).strip()
        
        if not param_str:
            raise ValueError(f"Parameter at index {i} is empty or only whitespace")
        
        # Check for dangerous characters that could lead to command injection
        # Disallow semicolons, pipes, backticks, dollar signs, and other shell metacharacters
        dangerous_chars = r'[;&|`$<>\\]'
        if re.search(dangerous_chars, param_str):
            raise ValueError(f"Parameter '{param_str}' contains potentially dangerous characters")
        
        command_parts.append(param_str)
    
    # Construct the full command string for validation
    full_command = ' '.join(command_parts)
    
    # Final validation of the complete command
    if len(full_command) > 1000:  # Arbitrary limit to prevent extremely long commands
        raise ValueError("Constructed command is too long (max 1000 characters)")
    
    try:
        # Execute the command
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed with return code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        raise ValueError(f"Command not found: {base_command}")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Test the function with valid inputs
    try:
        base = "echo"
        params = ["Hello", "World"]
        result = execute_command(base, params)
        print(f"Output: {result}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Another example
    try:
        base2 = "ls"
        params2 = ["-la"]
        result2 = execute_command(base2, params2)
        print(f"Output: {result2}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with invalid inputs
    try:
        base3 = "rm -rf /"
        params3 = ["; echo", "malicious"]
        result3 = execute_command(base3, params3)
        print(f"Output: {result3}")
    except ValueError as e:
        print(f"Error: {e}")  # Should raise error for semicolon
    
    # Test with empty base command
    try:
        base4 = ""
        params4 = ["test"]
        result4 = execute_command(base4, params4)
        print(f"Output: {result4}")
    except ValueError as e:
        print(f"Error: {e}")  # Should raise error for empty base command
    
    # Test with non-existent command
    try:
        base5 = "nonexistentcommand"
        params5 = ["--help"]
        result5 = execute_command(base5, params5)
        print(f"Output: {result5}")
    except ValueError as e:
        print(f"Error: {e}")  # Should raise error for command not found
