import subprocess
base_command = "echo Hello, {user_name}!"

def generate_greeting_command(user_name: str) -> str:
    """
    Generates a shell command to greet a user by name.
    
    Args:
        user_name: str, the user-provided name to be embedded into the command
        
    Returns:
        str, the generated shell command incorporating the user-provided name
        
    Note:
        This function does not sanitize the input, which could lead to command injection
        if user_name contains shell metacharacters.
    """
    return base_command.format(user_name=user_name)
