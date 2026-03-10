# List of dangerous keywords/patterns
# This list is not exhaustive and might need to be expanded for robust security.
_DANGEROUS_KEYWORDS = [
    "import", "__", "eval", "exec", "open", "os", "sys", "subprocess",
    "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "lambda", "input", "file", "socket", "urllib", "requests", "ctypes", "pty"
]

def _is_query_dangerous(query: str) -> bool:
    """Checks if the query contains any dangerous keywords."""
    # Using 'in' for substring check is basic; more robust parsing (e.g., AST) might be needed for complex obfuscation.
    sanitized_query = query.lower() # Check in lowercase to catch variations
    for keyword in _DANGEROUS_KEYWORDS:
        if keyword in sanitized_query:
            return True
    return False

def _evaluate_safe_query(query: str) -> str:
    """
    Evaluates a query string as a Python expression in a restricted environment.
    Propagates exceptions if evaluation fails.
    """
    # Evaluate the query.
    # Providing empty dictionaries for globals and locals restricts access to built-in functions and variables from the calling scope.
    # Only very basic expressions (like '2+2') or expressions involving literals will work.
    # To allow specific safe functions (e.g., 'abs', 'min', 'max'), they must be explicitly added to the globals_dict.
    # For example: safe_globals = {"abs": abs, "min": min, "max": max, "__builtins__": {}}
    # The current implementation uses completely empty globals and locals.
    result = eval(query, {"__builtins__": {}}, {}) # Restrict builtins
    return str(result)

def run_user_query(query: str) -> str:
    """
    Evaluates a user's query string as a Python expression.

    Args:
        query: A string containing the user's query.

    Returns:
        The result of the query evaluation as a string, or 'Error!'
        if there are issues during processing or dangerous input is detected.
    """
    if _is_query_dangerous(query):
        return 'Error!'

    try:
        result_str = _evaluate_safe_query(query)
        return result_str
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, MemoryError, RecursionError, Exception) as e:
        # Catching a broad range of common exceptions that might occur during evaluation.
        # Specific error handling or logging could be added here if needed.
        return 'Error!'

if __name__ == '__main__':
    # Example Usage for the implemented run_user_query:
    print(f"Query '2 + 2': {run_user_query('2 + 2')}")
    print(f"Query '10 / 2': {run_user_query('10 / 2')}")
    
    # These will result in 'Error!' because functions like 'abs' are not in the empty __builtins__
    print(f"Query 'abs(-5)': {run_user_query('abs(-5)')}") 
    
    # Dangerous queries - should return 'Error!' due to keyword checks
    print(f"Query 'import os': {run_user_query('import os')}")
    dangerous_query_import_os = '__import__("os").system("echo hacked")'
    print(f"Query '{dangerous_query_import_os}': {run_user_query(dangerous_query_import_os)}")
    dangerous_query_eval = 'eval("1+1")'
    print(f"Query '{dangerous_query_eval}': {run_user_query(dangerous_query_eval)}")
    dangerous_query_open_file = 'open("file.txt")'
    print(f"Query '{dangerous_query_open_file}': {run_user_query(dangerous_query_open_file)}")


    # Queries that would cause runtime errors, caught by try-except
    print(f"Query '1/0': {run_user_query('1/0')}") # ZeroDivisionError
    print(f"Query 'a + b': {run_user_query('a + b')}") # NameError (since 'a' and 'b' are not defined in the eval context)
    print(f"Query 'x = 5': {run_user_query('x = 5')}") # SyntaxError (eval cannot execute statements like assignments)
    
    # Example of a query that might be considered safe but uses a keyword that is blocked
    query_str_locals = '{"x": 1, "y": 2}.get("x", 0) + locals().get("z", 5)'
    print(f"Query '{query_str_locals}': {run_user_query(query_str_locals)}") # 'locals' is blocked

    # Example of a query that is too long / complex (might hit recursion or memory limits)
    # This is a simple arithmetic expression, should be fine.
    # For very complex ones, MemoryError or RecursionError might be caught.
    long_query = "+".join(["1"] * 1000) # "1+1+...+1" (1000 times)
    # print(f"Query '{long_query[:50]}...': {run_user_query(long_query)}") # This should evaluate correctly.

    # A query that might try to bypass keyword checks (though current check is basic)
    bypass_query_getattr = 'getattr(obj, "__im" + "port__")'
    print(f"Query '{bypass_query_getattr}': {run_user_query(bypass_query_getattr)}") # 'getattr' and '__' are blocked
