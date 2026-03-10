import ast

def run_user_query(query: str) -> str:
    """
    Evaluates a user's query string as a Python expression.

    Args:
        query: A string containing the user's query.

    Returns:
        The result of the query evaluation as a string, or 'Error!'
        if there are issues during processing or dangerous input is detected.
    """
    # Basic security check: Disallow common dangerous patterns or keywords.
    # This is not exhaustive and for a real-world application, a more robust sandboxing approach would be needed.
    dangerous_keywords = [
        "import",
        "exec",
        "eval",
        "open",
        "__",  # Disallow dunder methods/attributes to prevent access to internals
        "lambda", # Could be used to construct more complex calls
        "compile",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars"
    ]

    if any(keyword in query for keyword in dangerous_keywords):
        return "Error!"

    try:
        # Further check: Parse the expression and ensure it's simple.
        # We only allow simple expressions, not statements or complex constructs.
        # ast.Expression ensures it's an expression.
        # We can iterate through nodes to disallow certain node types if needed,
        # but for now, just ensuring it's a valid expression is a good step.
        parsed_expression = ast.parse(query, mode='eval')

        # Check for function calls, attribute access, etc., that might be risky
        # This is a more advanced check and can be expanded.
        # For now, we'll keep it simple and rely on the keyword blacklist and eval's limited scope.
        # A more robust solution might involve whitelisting allowed nodes or names.

        # Evaluate the expression.
        # Provide a limited scope for eval to run in.
        # No builtins are explicitly removed here, but for higher security,
        # one might provide a __builtins__ dict with only safe functions.
        result = eval(query, {"__builtins__": {}}, {}) # Empty globals, empty locals
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, AttributeError, KeyError, IndexError, ValueError, Exception):
        # Catch a broad range of exceptions that can occur during parsing or evaluation.
        return "Error!"

if __name__ == '__main__':
    # Example Usage:
    print(f"Query: '2 + 2', Result: {run_user_query('2 + 2')}")
    print(f"Query: '10 / 2', Result: {run_user_query('10 / 2')}")
    print(f"Query: 'abs(-5)', Result: {run_user_query('abs(-5)')}") # This will be "Error!" due to empty builtins
    
    # Example of a safe query that would work if builtins were selectively allowed
    # For example, if eval(query, {"__builtins__": {"abs": abs}}, {})
    # print(f"Query: 'abs(-5)', Result: {run_user_query('abs(-5)')}")


    # Dangerous inputs
    query1 = '__import__("os").system("echo pwned")'
    print(f"Query: '{query1}', Result: {run_user_query(query1)}")
    
    query2 = 'open("/etc/passwd").read()'
    print(f"Query: '{query2}', Result: {run_user_query(query2)}")
    
    print(f"Query: '1 / 0', Result: {run_user_query('1 / 0')}")
    print(f"Query: 'some_undefined_variable', Result: {run_user_query('some_undefined_variable')}")
    
    query3 = 'print("hello")' # print is a statement, not an expression for eval
    print(f"Query: '{query3}', Result: {run_user_query(query3)}") # and also not in the empty builtins
    
    query4 = '[x for x in ().__class__.__bases__[0].__subclasses__() if x.__name__ == "Quitter"][0](0, "you got pwned")()'
    print(f"Query: '{query4}', Result: {run_user_query(query4)}")
