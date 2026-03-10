# Assume COMMAND_VALUES is a globally available dictionary
# containing the placeholder keys and their corresponding values.
# For example:
# COMMAND_VALUES = {
#     "filename": "example.txt",
#     "directory": "/home/user/documents",
#     "options": "-al"
# }

import shlex

# This is a placeholder for the actual global structure.
# The user will need to ensure this is populated appropriately in their environment.
COMMAND_VALUES = {}

# Global base command for greeting
base_command = "echo Hello, {name_of_user}!"


def assemble_greet_command(name_of_user: str) -> str:
    """
    Builds a shell command to greet a user by their name using a predefined base command.

    Args:
        name_of_user: The name of the user to greet.

    Returns:
        The constructed shell command string.
    """
    quoted_name = shlex.quote(name_of_user)
    return base_command.format(name_of_user=quoted_name)


def insert_name_into_command(command_template: str, name_string: str) -> str:
    """
    Safely incorporates a name string into a command template.

    Args:
        command_template: A shell command template with a placeholder for the name (e.g., "echo Hello {name}").
                          The placeholder for the name should be {name}.
        name_string: The name string to be inserted into the command.

    Returns:
        A complete shell command string with the name safely incorporated.
    """
    quoted_name = shlex.quote(name_string)
    return command_template.format(name=quoted_name)


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

    # Example Usage for insert_name_into_command:
    name_template1 = "echo 'Hello, {name}!'"
    user_name1 = "Alice"
    named_command1 = insert_name_into_command(name_template1, user_name1)
    print(f"\nName Template 1: \"{name_template1}\" with name \"{user_name1}\"")
    print(f"Named Command 1: \"{named_command1}\"")
    # Expected: Named Command 1: "echo 'Hello, Alice!'" (or similar, depending on shlex.quote behavior for simple strings)

    name_template2 = "touch /tmp/{name}_file.txt"
    user_name2 = "Bob O'Malley" # Name with a space and apostrophe
    named_command2 = insert_name_into_command(name_template2, user_name2)
    print(f"\nName Template 2: \"{name_template2}\" with name \"{user_name2}\"")
    print(f"Named Command 2: \"{named_command2}\"")
    # Expected: Named Command 2: "touch /tmp/'Bob O'\\''Malley'_file.txt" (shlex.quote will handle special characters)

    name_template3 = "userdel {name}"
    user_name3 = "eve; rm -rf /" # Potentially malicious name
    named_command3 = insert_name_into_command(name_template3, user_name3)
    print(f"\nName Template 3: \"{name_template3}\" with name \"{user_name3}\"")
    print(f"Named Command 3: \"{named_command3}\"")
    # Expected: Named Command 3: "userdel 'eve; rm -rf /'" (shlex.quote prevents injection)

    # Example Usage for assemble_greet_command:
    greet_name1 = "World"
    greet_command1 = assemble_greet_command(greet_name1)
    print(f"\nGreeting for: \"{greet_name1}\"")
    print(f"Greet Command 1: \"{greet_command1}\"")
    # Expected: Greet Command 1: "echo Hello, World!"

    greet_name2 = "Charlie Brown"
    greet_command2 = assemble_greet_command(greet_name2)
    print(f"\nGreeting for: \"{greet_name2}\"")
    print(f"Greet Command 2: \"{greet_command2}\"")
    # Expected: Greet Command 2: "echo Hello, 'Charlie Brown'!"

    greet_name3 = "Dr. Evil; id"
    greet_command3 = assemble_greet_command(greet_name3)
    print(f"\nGreeting for: \"{greet_name3}\"")
    print(f"Greet Command 3: \"{greet_command3}\"")
    # Expected: Greet Command 3: "echo Hello, 'Dr. Evil; id'!"
