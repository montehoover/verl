def construct_command(base_command, parameters):
    """
    Constructs a full command string from a base command and a list of parameters.
    
    Args:
        base_command (str): The base command string
        parameters (list): List of parameters to append to the command
        
    Returns:
        str: The full command with all parameters concatenated
    """
    # Start with the base command
    full_command = base_command
    
    # Add each parameter separated by spaces
    for param in parameters:
        full_command += " " + str(param)
    
    return full_command


# Example usage
if __name__ == "__main__":
    # Test the function
    base = "git"
    params = ["commit", "-m", "Initial commit"]
    result = construct_command(base, params)
    print(result)  # Output: git commit -m Initial commit
    
    # Another example
    base2 = "python"
    params2 = ["script.py", "--verbose", "--output", "result.txt"]
    result2 = construct_command(base2, params2)
    print(result2)  # Output: python script.py --verbose --output result.txt
