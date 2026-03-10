def evaluate_expression(expr: str) -> str:
    try:
        # Restrict potentially dangerous operations
        if any(forbidden in expr for forbidden in ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'compile', 'globals', 'locals']):
            return 'Error!'
        
        # Create a restricted environment
        safe_dict = {
            '__builtins__': {
                'abs': abs,
                'all': all,
                'any': any,
                'bin': bin,
                'bool': bool,
                'chr': chr,
                'dict': dict,
                'divmod': divmod,
                'enumerate': enumerate,
                'filter': filter,
                'float': float,
                'format': format,
                'hex': hex,
                'int': int,
                'len': len,
                'list': list,
                'map': map,
                'max': max,
                'min': min,
                'oct': oct,
                'ord': ord,
                'pow': pow,
                'range': range,
                'reversed': reversed,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
            }
        }
        
        # Evaluate the expression
        result = eval(expr, safe_dict, {})
        return str(result)
    except:
        return 'Error!'
