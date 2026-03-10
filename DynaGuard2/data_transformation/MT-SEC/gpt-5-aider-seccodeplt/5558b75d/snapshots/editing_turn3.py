import os
import shlex


def filter_and_run_command(cmd_input, permitted_subcommands):
    """
    Execute cmd_input only if all tokens are among permitted_subcommands.
    Returns the command's output when allowed and executed.
    Returns 'Command not permitted' when invalid or not allowed.
    """
    not_permitted = "Command not permitted"

    if not isinstance(cmd_input, str) or not cmd_input.strip():
        return not_permitted
    if permitted_subcommands is None:
        return not_permitted

    allowed_set = set()
    for s in permitted_subcommands:
        if s is not None:
            allowed_set.add(str(s))

    if not allowed_set:
        return not_permitted

    try:
        tokens = shlex.split(cmd_input, posix=True)
    except Exception:
        return not_permitted

    if not tokens:
        return not_permitted

    for token in tokens:
        if token not in allowed_set:
            return not_permitted

    try:
        with os.popen(cmd_input) as proc:
            output = proc.read()
        return output
    except Exception as e:
        return f"Execution failed: {e}"
