import ast

def execute_user_code(script_code: str):
    """
    Evaluates a user-supplied Python script string, allowing only basic
    arithmetic and string operations.

    Args:
        script_code: A string containing the Python script to evaluate.
                     Supported arithmetic operations: +, -, *, /.
                     Supported string operations: + (concatenation), * (repetition).

    Returns:
        The result of the script execution, or 'Execution Blocked!' if the
        script is unsafe or contains invalid operations.
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
        # Parse the script code into an AST (Abstract Syntax Tree)
        # mode='eval' is used for a single expression
        node = ast.parse(script_code, mode='eval')

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
    print(f"Testing '2 + 3 * 4': {execute_user_code('2 + 3 * 4')}")  # Expected: 14
    print(f"Testing '10 / 2 - 1': {execute_user_code('10 / 2 - 1')}")  # Expected: 4.0
    print(f"Testing '(2 + 3) * 4': {execute_user_code('(2 + 3) * 4')}") # Expected: 20
    print(f"Testing '10 / 0': {execute_user_code('10 / 0')}")          # Expected: Execution Blocked!
    print(f"Testing 'abs(-5)': {execute_user_code('abs(-5)')}")        # Expected: Execution Blocked! (abs is not allowed)
    unsafe_expr_str = '__import__("os").system("echo pwned")'
    print(f"Testing '__import__(\"os\").system(\"echo pwned\")': {execute_user_code(unsafe_expr_str)}") # Expected: Execution Blocked!
    print(f"Testing '2 ** 3': {execute_user_code('2 ** 3')}")          # Expected: Execution Blocked! (Power operator not explicitly allowed)
    print(f"Testing 'a = 5; a + 2': {execute_user_code('a = 5; a + 2')}") # Expected: Execution Blocked! (Assignment not allowed in 'eval' mode)
    print(f"Testing '1 + ': {execute_user_code('1 + ')}")              # Expected: Execution Blocked! (SyntaxError)
    print(f"Testing '-5 + 2': {execute_user_code('-5 + 2')}")          # Expected: -3
    print(f"Testing '3 * -2': {execute_user_code('3 * -2')}")          # Expected: -6
    print(f"Testing '1.5 * 2': {execute_user_code('1.5 * 2')}")        # Expected: 3.0
    print(f"Testing 'foo': {execute_user_code('foo')}")                # Expected: Execution Blocked! (NameError)
    
    # String operation test cases
    expr_str_concat = "'hello' + ' ' + 'world'"
    print(f"Testing \"'hello' + ' ' + 'world'\": {execute_user_code(expr_str_concat)}") # Expected: hello world
    
    expr_str_repeat = "'foo' * 3"
    print(f"Testing \"'foo' * 3\": {execute_user_code(expr_str_repeat)}")      # Expected: foofoofoo
    
    expr_str_type_error = "3 + 'foo'"
    print(f"Testing \"3 + 'foo'\": {execute_user_code(expr_str_type_error)}")        # Expected: Execution Blocked! (TypeError)
    
    expr_str_unsupported_op = "'foo' - 'o'"
    print(f"Testing \"'foo' - 'o'\": {execute_user_code(expr_str_unsupported_op)}")      # Expected: Execution Blocked! (TypeError)
