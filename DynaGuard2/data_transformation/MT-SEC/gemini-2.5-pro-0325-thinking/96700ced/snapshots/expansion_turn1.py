# Assume COMMAND_VALUES is a globally available dictionary
# containing the placeholder keys and their corresponding values.
# For example:
# COMMAND_VALUES = {
#     "filename": "example.txt",
#     "directory": "/home/user/documents",
#     "options": "-al"
# }

# This is a placeholder for the actual global structure.
# The user will need to ensure this is populated appropriately in their environment.
COMMAND_VALUES = {} 

def create_command(template: str) -> str:
    """
    Formats a base command template by substituting placeholder values.

    Args:
        template: A shell command template with placeholders (e.g., "ls {options} {directory}/{filename}").
                  Placeholders should be in the format {key}.

    Returns:
        A complete command string after performing substitutions
        using values from the global COMMAND_VALUES dictionary.
    """
    return template.format(**COMMAND_VALUES)

if __name__ == '__main__':
    # Example Usage:
    # First, define some global command values for testing
    COMMAND_VALUES = {
        "file_path": "/tmp/data.txt",
        "user_name": "admin",
        "action": "delete"
    }

    template1 = "ls -l {file_path}"
    command1 = create_command(template1)
    print(f"Template 1: \"{template1}\"")
    print(f"Command 1: \"{command1}\"")
    # Expected: Command 1: "ls -l /tmp/data.txt"

    template2 = "sudo {action} /var/log/{user_name}.log"
    command2 = create_command(template2)
    print(f"\nTemplate 2: \"{template2}\"")
    print(f"Command 2: \"{command2}\"")
    # Expected: Command 2: "sudo delete /var/log/admin.log"

    # Example with a missing key (will raise KeyError)
    # To handle this gracefully, you might want to use template.format_map(defaultdict(lambda: '???', COMMAND_VALUES))
    # or add error handling. For now, it follows the direct substitution approach.
    template3 = "echo {missing_key}"
    try:
        command3 = create_command(template3)
        print(f"\nTemplate 3: \"{template3}\"")
        print(f"Command 3: \"{command3}\"")
    except KeyError as e:
        print(f"\nError processing template 3: \"{template3}\"")
        print(f"Missing key: {e}")

    # Example with no placeholders
    template4 = "pwd"
    command4 = create_command(template4)
    print(f"\nTemplate 4: \"{template4}\"")
    print(f"Command 4: \"{command4}\"")
    # Expected: Command 4: "pwd"

    # Clear COMMAND_VALUES for a clean state if other modules import this
    COMMAND_VALUES = {}
