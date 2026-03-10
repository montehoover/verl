import ast

def calculate_expression(expression: str):
    """
    Calculates the result of a simple arithmetic expression string.

    Args:
        expression: A string containing a simple arithmetic expression.
                    Supported operations: +, -, *, /.

    Returns:
        The result of the expression, or 'Execution Blocked!' if the
        expression is invalid or unsafe.
    """
    allowed_nodes = {
        ast.Expression,
        ast.Num,  # Deprecated in Python 3.8, use ast.Constant
        ast.Constant, # For numbers and strings (though we only expect numbers)
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub, # For negative numbers
        ast.Load, # Context for loading a variable (not used directly but part of AST)
    }

    try:
        # Parse the expression into an AST (Abstract Syntax Tree)
        # mode='eval' is used for a single expression
        node = ast.parse(expression, mode='eval')

        # Validate the AST nodes
        for sub_node in ast.walk(node):
            if not isinstance(sub_node, tuple(allowed_nodes)):
                # Disallow any node type not in our whitelist
                return 'Execution Blocked!'
        
        # If all nodes are allowed, compile and evaluate the expression
        # We use a restricted globals and locals dictionary for eval for added safety,
        # though the AST validation should prevent malicious code.
        compiled_code = compile(node, filename='<string>', mode='eval')
        result = eval(compiled_code, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, TypeError, ZeroDivisionError, NameError, KeyError, OverflowError, ValueError, AttributeError):
        # Catch common errors during parsing or evaluation
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected errors
        return 'Execution Blocked!'

if __name__ == '__main__':
    # Test cases
    print(f"Testing '2 + 3 * 4': {calculate_expression('2 + 3 * 4')}")  # Expected: 14
    print(f"Testing '10 / 2 - 1': {calculate_expression('10 / 2 - 1')}")  # Expected: 4.0
    print(f"Testing '(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}") # Expected: 20
    print(f"Testing '10 / 0': {calculate_expression('10 / 0')}")          # Expected: Execution Blocked!
    print(f"Testing 'abs(-5)': {calculate_expression('abs(-5)')}")        # Expected: Execution Blocked! (abs is not allowed)
    unsafe_expr_str = '__import__("os").system("echo pwned")'
    print(f"Testing '__import__(\"os\").system(\"echo pwned\")': {calculate_expression(unsafe_expr_str)}") # Expected: Execution Blocked!
    print(f"Testing '2 ** 3': {calculate_expression('2 ** 3')}")          # Expected: Execution Blocked! (Power operator not explicitly allowed)
    print(f"Testing 'a = 5; a + 2': {calculate_expression('a = 5; a + 2')}") # Expected: Execution Blocked! (Assignment not allowed in 'eval' mode)
    print(f"Testing '1 + ': {calculate_expression('1 + ')}")              # Expected: Execution Blocked! (SyntaxError)
    print(f"Testing '-5 + 2': {calculate_expression('-5 + 2')}")          # Expected: -3
    print(f"Testing '3 * -2': {calculate_expression('3 * -2')}")          # Expected: -6
    print(f"Testing '1.5 * 2': {calculate_expression('1.5 * 2')}")        # Expected: 3.0
    print(f"Testing 'foo': {calculate_expression('foo')}")                # Expected: Execution Blocked! (NameError)
