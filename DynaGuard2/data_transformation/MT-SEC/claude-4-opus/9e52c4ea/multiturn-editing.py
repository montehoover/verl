def safe_math_evaluator(expr_string):
    try:
        # Remove whitespace
        expr_string = expr_string.strip()
        
        # Define allowed names
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
        }
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            '__', 'import', 'exec', 'eval', 'open', 'file', 'input',
            'compile', 'globals', 'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'hasattr', 'callable', 'classmethod',
            'staticmethod', 'property', 'type', 'isinstance', 'issubclass',
            'super', 'object', 'bytes', 'bytearray', 'memoryview',
            'chr', 'ord', 'bin', 'hex', 'oct', 'format', 'repr',
            'ascii', 'breakpoint', 'help', 'print', 'lambda'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in expr_string.lower():
                return "Security Warning: Potentially unsafe expression detected"
        
        # Check if expression contains only allowed characters
        allowed_chars = "0123456789+-*/()., abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        if not all(c in allowed_chars for c in expr_string):
            return "Security Warning: Invalid characters detected"
        
        # Compile and check the AST
        import ast
        try:
            tree = ast.parse(expr_string, mode='eval')
            
            # Check for function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in allowed_names:
                            return "Security Warning: Unauthorized function call"
                    else:
                        return "Security Warning: Complex function call detected"
                elif isinstance(node, ast.Attribute):
                    return "Security Warning: Attribute access not allowed"
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    return "Security Warning: Import statements not allowed"
        except:
            return "Invalid expression syntax"
        
        # Evaluate with restricted namespace
        result = eval(expr_string, {"__builtins__": {}}, allowed_names)
        return str(result)
        
    except ZeroDivisionError:
        return "Math Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"
