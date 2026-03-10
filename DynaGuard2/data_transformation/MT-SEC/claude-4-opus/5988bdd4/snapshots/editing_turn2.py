import sys
from io import StringIO

def evaluate_expression(expression):
    """Evaluate a Python expression or execute code and return the result/output."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Try to evaluate as an expression first
        result = eval(expression)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        if output:
            return output.rstrip()
        else:
            return result
    except:
        # If eval fails, try exec for statements
        sys.stdout = StringIO()
        try:
            exec(expression)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            return output.rstrip() if output else None
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {e}"
