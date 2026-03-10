import ast

# Mapping of AST binary operator types to their corresponding functions
_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

# Mapping of AST unary operator types to their corresponding functions
_UNARY_OPERATORS = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}

def _evaluate_ast_node(node):
    """
    Recursively evaluates an AST node that has already been validated.
    Assumes the node and its children conform to the allowed subset of Python grammar.
    """
    if isinstance(node, ast.Expression):
        # The top-level node for an expression parsed in 'eval' mode.
        # Evaluate its body.
        return _evaluate_ast_node(node.body)
    elif isinstance(node, ast.Constant):
        # Value has been validated to be an int or float.
        return node.value
    elif isinstance(node, ast.UnaryOp):
        # Operator type (node.op) has been validated by the main function's walk.
        operand_value = _evaluate_ast_node(node.operand)
        return _UNARY_OPERATORS[type(node.op)](operand_value)
    elif isinstance(node, ast.BinOp):
        # Operator type (node.op) has been validated by the main function's walk.
        left_value = _evaluate_ast_node(node.left)
        right_value = _evaluate_ast_node(node.right)
        return _OPERATORS[type(node.op)](left_value, right_value)
    else:
        # This path should not be reached if validation was thorough
        # and the AST corresponds to a simple mathematical expression.
        raise ValueError(f"Unsupported AST node type encountered during evaluation: {type(node)}")

def _validate_ast(tree: ast.AST) -> bool:
    """
    Validates if all nodes in the AST are allowed for safe evaluation.
    Returns True if safe, False otherwise.
    """
    # ast.walk yields all nodes in the tree, including operator types and context markers.
    for node in ast.walk(tree):
        node_type = type(node)

        if node_type is ast.Expression:
            # Root node for 'eval' mode, always allowed. Its body is checked.
            continue
        elif node_type is ast.Constant:
            # Allow only numeric constants (int or float).
            if not isinstance(node.value, (int, float)):
                return False
        elif node_type is ast.BinOp:
            # Binary operation node. Its operator (node.op) must be an allowed type.
            # The operator type itself (e.g., ast.Add) will also be visited by ast.walk.
            if type(node.op) not in _OPERATORS:
                return False
        elif node_type is ast.UnaryOp:
            # Unary operation node. Its operator (node.op) must be an allowed type.
            if type(node.op) not in _UNARY_OPERATORS:
                return False
        # Check for allowed operator types themselves (e.g., ast.Add, ast.Sub).
        # These are yielded by ast.walk when traversing BinOp.op or UnaryOp.op.
        elif node_type in _OPERATORS or node_type in _UNARY_OPERATORS:
            continue
        # Allow context markers like ast.Load. These are part of valid AST structure
        # but don't represent operations themselves.
        # ast.Store and ast.Del are typically not needed for simple expression evaluation
        # but are included here for completeness if the grammar allows them in 'eval' mode
        # for some Python versions/constructs, though our rules should prevent their misuse.
        # For strictness, one might remove ast.Store and ast.Del if not strictly necessary.
        elif node_type in (ast.Load, ast.Store, ast.Del): # ast.Load is essential.
            continue
        else:
            # Any other node type (e.g., ast.Name, ast.Call, ast.Attribute, ast.IfExp)
            # is considered unsafe for this simple math evaluator.
            return False
    return True

def evaluate_math_expression(math_expression: str) -> str:
    """
    Securely evaluates a string containing mathematical expressions.
    Avoids direct use of eval() or exec().
    Returns the result as a string or 'Unsafe Code Detected!' on error or unsafe content.
    """
    try:
        # 1. Parse the expression string into an AST.
        #    mode='eval' ensures the input is a single expression.
        tree = ast.parse(math_expression, mode='eval')

        # 2. Validate the AST.
        if not _validate_ast(tree):
            return "Unsafe Code Detected!"
        
        # 3. If the AST is valid, evaluate it.
        #    The `tree` variable holds an ast.Expression node.
        result = _evaluate_ast_node(tree)
        
        # Ensure the result is a number (int or float) before converting to string.
        # This is a final safeguard, as _evaluate_ast_node should only produce numbers
        # if the AST was correctly validated and only contains allowed numeric operations.
        if not isinstance(result, (int, float)):
            # This might indicate an issue in _evaluate_ast_node or _validate_ast
            # if it's reached with a valid-looking expression.
            return "Unsafe Code Detected!"

        return str(result)

    except SyntaxError:
        # The input string is not valid Python syntax.
        return "Unsafe Code Detected!"
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        # Errors during evaluation (e.g., division by zero, math domain errors).
        # ValueError can also be raised by _evaluate_ast_node for unhandled validated nodes.
        return "Unsafe Code Detected!"
    except RecursionError: 
        # Expression too complex, causing too deep recursion.
        return "Unsafe Code Detected!"
    except Exception:
        # Catch-all for any other unexpected errors.
        return "Unsafe Code Detected!"

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print(f"'2 + 2': {evaluate_math_expression('2 + 2')}")  # Expected: 4
    print(f"'10 / 2': {evaluate_math_expression('10 / 2')}") # Expected: 5.0
    print(f"'3 * (4 + 2)': {evaluate_math_expression('3 * (4 + 2)')}") # Expected: 18
    print(f"'2 ** 3': {evaluate_math_expression('2 ** 3')}") # Expected: 8
    print(f"'-5': {evaluate_math_expression('-5')}") # Expected: -5
    print(f"'+5': {evaluate_math_expression('+5')}") # Expected: 5
    print(f"'1 / 0': {evaluate_math_expression('1 / 0')}") # Expected: Unsafe Code Detected!
    print(f"'abs(5)': {evaluate_math_expression('abs(5)')}") # Expected: Unsafe Code Detected!
    print(f"'x = 5': {evaluate_math_expression('x = 5')}") # Expected: Unsafe Code Detected! (SyntaxError)
    test_expr_import_os = """__import__("os").system("clear")"""
    print(f"'{test_expr_import_os}': {evaluate_math_expression(test_expr_import_os)}") # Expected: Unsafe Code Detected!
    print(f"'1 + 1.0': {evaluate_math_expression('1 + 1.0')}") # Expected: 2.0
    test_expr_hello_world = '''"hello" + "world"'''
    print(f"'{test_expr_hello_world}': {evaluate_math_expression(test_expr_hello_world)}") # Expected: Unsafe Code Detected! (string constant)
    print(f"'(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+1))))))))))': {evaluate_math_expression('(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+1))))))))))')}") # Deeply nested, test recursion
    # Test for very long expression that might hit recursion limits (actual limit depends on system)
    # This one is just for illustration, actual recursion depth for failure varies.
    # long_expr = "+".join(["1"] * 2000) # Python's AST parser might handle this differently than recursive eval
    # print(f"Long expression: {evaluate_math_expression(long_expr)}")
    print(f"Empty string '': {evaluate_math_expression('')}") # Expected: Unsafe Code Detected!
    print(f"Only spaces '   ': {evaluate_math_expression('   ')}") # Expected: Unsafe Code Detected!
    print(f"Invalid syntax '1 +': {evaluate_math_expression('1 +')}") # Expected: Unsafe Code Detected!
    print(f"Boolean 'True': {evaluate_math_expression('True')}") # Expected: Unsafe Code Detected!
