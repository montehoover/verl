import subprocess
import shlex

def execute_shell_command(command: str, args: list[str]) -> str:
    try:
        # Build the full command with arguments
        full_command = [command] + args
        
        # Execute the command
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        
        # Return the output as a string
        return result.stdout.strip()
            
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to execute command: {command}")
