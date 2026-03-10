import subprocess
import shlex

def execute_shell_command(command: str) -> str:
    try:
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        # Return the output as a string
        return result.stdout.strip()
            
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to execute command: {command}")
