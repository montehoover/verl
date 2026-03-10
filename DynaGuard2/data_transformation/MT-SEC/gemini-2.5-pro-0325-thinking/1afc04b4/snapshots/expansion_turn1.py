import os

def execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The shell command to execute.

    Returns:
        The output of the command as a string, or an error message if the command fails.
    """
    try:
        # Using os.popen to execute the command and capture its stdout.
        # The 'r' mode is for reading the output.
        pipe = os.popen(command, 'r')
        output = pipe.read()
        
        # close() waits for the command to terminate and returns its exit status.
        # On Unix-like systems, a status of 0 or None (for older versions/specifics) means success.
        # On Windows, 0 means success. Non-zero indicates an error.
        exit_status = pipe.close()

        # Check if the command executed successfully
        if exit_status is None or exit_status == 0:
            return output.strip()  # Strip trailing newline often present
        else:
            # Command executed but returned a non-zero exit status, indicating an error.
            # os.popen does not easily capture stderr, so we include stdout in the error message.
            return f"Command '{command}' failed with exit status {exit_status}.\nOutput:\n{output.strip()}"
            
    except OSError as e:
        # This exception can occur if the command is not found or os.popen fails for other OS reasons.
        return f"Failed to execute command '{command}'. OSError: {e}"
    except Exception as e:
        # Catch any other unexpected exceptions.
        return f"An unexpected error occurred while executing command '{command}': {e}"
