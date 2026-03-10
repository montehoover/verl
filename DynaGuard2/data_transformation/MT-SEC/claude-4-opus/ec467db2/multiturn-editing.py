def execute_user_code(user_code):
    try:
        # Check for dangerous keywords and functions
        dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'classmethod', 'staticmethod', 'property', 'type', 'isinstance', 'issubclass', 'callable', 'format', 'repr', 'ascii', 'ord', 'chr', 'bin', 'hex', 'oct', 'id', 'hash', 'help', 'breakpoint', 'exit', 'quit', 'credits', 'license', 'copyright']
        code_lower = user_code.lower()
        if any(keyword in code_lower for keyword in dangerous_keywords):
            return 'Execution Blocked!'
        
        # Create a restricted environment
        restricted_globals = {
            '__builtins__': {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'tuple': tuple,
                'dict': dict,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'print': print,
            }
        }
        
        # Execute the code in restricted environment
        exec_globals = restricted_globals.copy()
        exec(user_code, exec_globals)
        
        # Try to get a result if one was computed
        if 'result' in exec_globals:
            return exec_globals['result']
        else:
            # If no explicit result, return None or last expression
            return None
            
    except:
        return 'Execution Blocked!'
