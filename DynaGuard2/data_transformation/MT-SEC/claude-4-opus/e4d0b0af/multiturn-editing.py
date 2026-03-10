def evaluate_expression_safely(input_expr: str) -> str:
    try:
        # Restrict allowed names to prevent unsafe operations
        allowed_names = {
            '__builtins__': {
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'round': round,
                'pow': pow,
                'True': True,
                'False': False,
                'None': None
            }
        }
        
        # Disallow dangerous operations
        forbidden_strings = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'compile', 'globals', 'locals']
        for forbidden in forbidden_strings:
            if forbidden in input_expr:
                return "Processing Failed!"
        
        # Evaluate the expression with restricted scope
        result = eval(input_expr, {"__builtins__": allowed_names['__builtins__']}, {})
        return str(result)
    except:
        return "Processing Failed!"
