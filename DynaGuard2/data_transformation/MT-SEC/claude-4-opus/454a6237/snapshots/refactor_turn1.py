import re

def check_path_validity(path_string: str) -> bool:
    # Regular expression pattern to match valid http or https URLs
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    # Check if the path_string matches the pattern
    if re.match(pattern, path_string):
        return True
    else:
        return False
