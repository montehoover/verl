import shlex

def sanitize_input(user_input):
    """
    Sanitize user input to prevent shell command injection vulnerabilities.
    
    Args:
        user_input (str): The user input string to sanitize
        
    Returns:
        str: Safely escaped version of the input
    """
    # Use shlex.quote() to properly escape the input for shell usage
    # This adds quotes around the string and escapes any special characters
    return shlex.quote(user_input)


# Example usage:
if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "normal input",
        "input with spaces",
        "input; rm -rf /",
        "input && echo 'hacked'",
        "input | cat /etc/passwd",
        "input`whoami`",
        "input$(whoami)",
        "input'with'quotes",
        'input"with"double"quotes',
        "input\\with\\backslashes",
        "input\nwith\nnewlines",
        "input\twith\ttabs"
    ]
    
    print("Testing sanitize_input function:")
    print("-" * 50)
    for test in test_inputs:
        sanitized = sanitize_input(test)
        print(f"Original: {repr(test)}")
        print(f"Sanitized: {sanitized}")
        print("-" * 50)
