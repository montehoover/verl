import subprocess
import shlex

def call_system_command(sys_command, params):
    try:
        command = [sys_command]
        if params:
            command.extend(params)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        raise ValueError("Command execution failed")
