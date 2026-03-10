def evaluate_expression(expr):
    if not expr or not isinstance(expr, str):
        return "Error!"
    
    # Check for potentially dangerous characters/patterns
    dangerous_patterns = ['import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input']
    
    try:
        if expr.startswith("DOUBLE:"):
            content = expr[7:]  # Remove "DOUBLE:" prefix
            # Check for dangerous patterns in content
            for pattern in dangerous_patterns:
                if pattern in content.lower():
                    return "Error!"
            return content + content
        
        elif expr.startswith("PALINDROME:"):
            content = expr[11:]  # Remove "PALINDROME:" prefix
            # Check for dangerous patterns in content
            for pattern in dangerous_patterns:
                if pattern in content.lower():
                    return "Error!"
            # Check if palindrome (case-insensitive)
            cleaned = content.lower().replace(" ", "")
            return str(cleaned == cleaned[::-1])
        
        else:
            # Default behavior - convert to lowercase
            expr_lower = expr.lower()
            for pattern in dangerous_patterns:
                if pattern in expr_lower:
                    return "Error!"
            return expr_lower
            
    except:
        return "Error!"
