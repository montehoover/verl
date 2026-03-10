def process_user_query(user_input: str) -> str:
    try:
        # Check if input contains potentially dangerous operations
        dangerous_keywords = ['__import__', 'exec', 'eval', 'open', 'file', 'input', 'compile', 'globals', 'locals', '__', 'import']
        for keyword in dangerous_keywords:
            if keyword in user_input:
                return 'Error!'
        
        # Evaluate the expression
        result = eval(user_input)
        return str(result)
    except:
        return 'Error!'
