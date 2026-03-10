def manage_script(user_script: str) -> str:
    # Check for suspicious keywords
    suspicious_keywords = ['exec', 'eval']
    for keyword in suspicious_keywords:
        if keyword in user_script:
            raise ValueError(f"Script contains suspicious keyword: {keyword}")
    
    # Try to compile the script to check if it's malformed
    try:
        compile(user_script, '<string>', 'exec')
    except SyntaxError:
        raise ValueError("Script is malformed")
    
    # Append the custom comment
    processed_script = user_script + "\n# Processed"
    
    return processed_script
