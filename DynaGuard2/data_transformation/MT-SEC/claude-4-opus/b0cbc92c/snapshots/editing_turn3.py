import subprocess
import shlex

def run_command(cmd, cmd_arguments):
    if not cmd or not isinstance(cmd, str):
        raise ValueError("Command must be a non-empty string")
    
    try:
        # Build the full command
        full_command = [cmd] + [str(arg) for arg in cmd_arguments]
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e}")
    except Exception as e:
        raise ValueError(f"Error processing command: {e}")
