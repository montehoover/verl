import json

def construct_command(main_command: str, args: list) -> str:
    """
    Constructs a command string for execution in a JavaScript console.

    Args:
        main_command: The main command string.
        args: A list of arguments. Arguments will be JSON-encoded if they are strings,
              otherwise converted to their string representation.

    Returns:
        A properly formatted command string.
    """
    if not isinstance(main_command, str):
        raise TypeError("main_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")

    formatted_args = []
    for arg in args:
        if isinstance(arg, str):
            # Use json.dumps to handle quoting and escaping for strings
            formatted_args.append(json.dumps(arg))
        elif isinstance(arg, (int, float, bool)):
            # For numbers and booleans, convert to string (JavaScript will parse them)
            formatted_args.append(str(arg).lower() if isinstance(arg, bool) else str(arg))
        else:
            # For other types, convert to string and wrap in quotes as a fallback
            # This might not be safe for all types, but covers simple cases.
            # Consider raising an error or having more specific handling if needed.
            formatted_args.append(json.dumps(str(arg)))

    return f"{main_command}({', '.join(formatted_args)})"
