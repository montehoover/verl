import subprocess
import shlex

def execute_shell_command(directory: str) -> None:
    try:
        # Execute ls command
        cmd = f"ls {shlex.quote(directory)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        # Parse output and format as comma-separated string
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]  # Remove empty strings
        
        if files:
            print(','.join(files))
        else:
            print("")
            
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to list directory: {directory}")
