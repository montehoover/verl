import re
import operator
import ast

# Supported operators mapping to functions
_SUPPORTED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    # Add more operators here if needed, e.g., ast.Pow for exponentiation
}

def _recursive_eval_ast_node(node, variable_mapping):
    """
    Recursively evaluates an AST node.
    """
    if isinstance(node, ast.Constant):
        # In Python 3.8+, strings and bytes are also ast.Constant.
        # We only want to process numbers here.
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value)}. Only numeric constants are allowed.")
        return node.value
    elif isinstance(node, ast.Name):
        if node.id not in variable_mapping:
            raise ValueError(f"Undefined variable: {node.id}")
        value = variable_mapping[node.id]
        # Ensure variables resolve to numbers
        if not isinstance(value, (int, float)):
            raise ValueError(f"Variable '{node.id}' must resolve to a number, got {type(value)}")
        return value
    elif isinstance(node, ast.BinOp):
        left_val = _recursive_eval_ast_node(node.left, variable_mapping)
        right_val = _recursive_eval_ast_node(node.right, variable_mapping)
        op_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        try:
            return op_func(left_val, right_val)
        except ZeroDivisionError:
            raise ValueError("Division by zero")
    elif isinstance(node, ast.UnaryOp):
        operand_val = _recursive_eval_ast_node(node.operand, variable_mapping)
        op_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand_val)
    else:
        # Catch any other node types (like ast.Call, ast.Attribute, ast.Subscript, etc.)
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

def _parse_to_ast_body(math_expression: str) -> ast.AST:
    """
    Parses a mathematical expression string into an AST body.
    Raises ValueError for syntax errors or invalid AST structure.
    """
    try:
        # Parse the expression in 'eval' mode, which expects a single expression
        parsed_ast = ast.parse(math_expression, mode='eval')

        # The body of an 'eval' mode AST is an ast.Expression node,
        # its 'body' attribute contains the actual expression node.
        if not isinstance(parsed_ast, ast.Expression) or not hasattr(parsed_ast, 'body'):
            # This case should ideally not be reached if ast.parse(mode='eval') works as expected
            # and the expression is valid.
            raise ValueError("Invalid expression structure after parsing.")
        return parsed_ast.body
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e.msg} at line {e.lineno}, offset {e.offset}")
    # Other exceptions during ast.parse are less common for mode='eval' if syntax is okay,
    # but if they occur, they would be caught by the generic Exception handler in the caller.

def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Evaluates a mathematical expression string with variable substitution using AST.

    Args:
        math_expression (str): A string representing a mathematical expression.
        variable_mapping (dict): A dictionary where keys are variable names
                                 and values are their corresponding numbers.

    Returns:
        str: The result of the simplified expression as a string.

    Raises:
        ValueError: If the input expression is invalid, contains unsupported
                    operations/constructs, or the computation fails.
    """
    if not isinstance(math_expression, str):
        raise ValueError("Expression must be a string.")
    if not isinstance(variable_mapping, dict):
        raise ValueError("Variable mapping must be a dictionary.")

    try:
        # Validate variable names and their values in the mapping
        for var_name, var_value in variable_mapping.items():
            if not isinstance(var_name, str) or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                raise ValueError(f"Invalid variable name: '{var_name}'")
            if not isinstance(var_value, (int, float)):
                raise ValueError(f"Variable '{var_name}' must be a number, got {type(var_value).__name__}")

        # Pipeline Step 1: Parse the expression string to an AST body.
        # This step handles SyntaxError and basic AST structure validation, raising ValueError on issues.
        ast_body = _parse_to_ast_body(math_expression)

        # Pipeline Step 2: Evaluate the AST body using the variable mapping.
        # This step handles undefined variables, unsupported operations, division by zero, etc.,
        # raising ValueError on issues.
        result = _recursive_eval_ast_node(ast_body, variable_mapping)
        
        # Ensure the result from evaluation is a number before converting to string
        if not isinstance(result, (int, float)):
            # This might indicate an issue with _recursive_eval_ast_node logic or an unsupported operation
            # that somehow passed through without raising a more specific ValueError earlier.
            raise ValueError("Evaluation did not result in a number.")
            
        return str(result)

    # ValueError can be raised by our input/variable checks, 
    # by _parse_to_ast_body (e.g., syntax errors, invalid AST structure),
    # or by _recursive_eval_ast_node (e.g., undefined variable, unsupported operator, math errors).
    except ValueError:
        raise
    # Catch any other unexpected errors during the process
    except Exception as e:
        # This is a fallback for truly unexpected issues not covered by ValueError.
        raise ValueError(f"Failed to evaluate expression due to an unexpected error: {type(e).__name__} - {e}")

if __name__ == '__main__':
    # Example Usage:
    print("Running examples...")
    try:
        # Basic arithmetic
        print(f"1 + 2 = {evaluate_expression_safely('1 + 2', {})}")
        print(f"5 - 3 * 2 = {evaluate_expression_safely('5 - 3 * 2', {})}")
        print(f"10 / 2 = {evaluate_expression_safely('10 / 2', {})}")
        print(f"-5 + 2 = {evaluate_expression_safely('-5 + 2', {})}")

        # With variables
        variables = {'x': 5, 'y': 10.5, '_z': 2}
        print(f"x + y with {variables} = {evaluate_expression_safely('x + y', variables)}")
        print(f"x * (y - _z) with {variables} = {evaluate_expression_safely('x * (y - _z)', variables)}")

        # Parentheses
        print(f"(1 + 2) * 3 = {evaluate_expression_safely('(1 + 2) * 3', {})}")

        # Floating point precision
        print(f"0.1 + 0.2 = {evaluate_expression_safely('0.1 + 0.2', {})}")

        # Unary minus
        print(f"-(3 + 2) = {evaluate_expression_safely('-(3+2)', {})}")
        print(f"10 + -2 = {evaluate_expression_safely('10 + -2', {})}")
        print(f"1_000_000 = {evaluate_expression_safely('1_000_000', {})}")
        print(f"1.5e2 = {evaluate_expression_safely('1.5e2', {})}")


    except ValueError as e:
        print(f"Error in basic examples: {e}")

    print("\nTesting error cases...")
    # (Expression, variable_map, expected_error_message_part)
    test_cases_fail = [
        ("1 / 0", {}, "Division by zero"),
        ("1 + z", {'x': 1}, "Undefined variable: z"),
        ("import os", {}, "Invalid syntax in expression"),
        ("x()", {'x':1}, "Unsupported expression construct: Call"),
        ("x[0]", {'x':[1,2]}, "Unsupported expression construct: Subscript"),
        ("1 & 2", {}, "Unsupported binary operator: BitAnd"),
        ("1 + y", {'y': 'text'}, "Variable 'y' must be a number, got str"),
        ("lambda: 1", {}, "Invalid syntax in expression"),
        ("a ? b : c", {}, "Invalid syntax in expression"), # Not Python syntax
        ("x + y", {"x": "foo", "y": 1}, "Variable 'x' must be a number, got str"),
        ("x + y", {"bad var name": 1, "y": 2}, "Invalid variable name: 'bad var name'"),
        ("1 + ", {}, "Invalid syntax in expression"),
        ("eval('1+1')", {}, "Unsupported expression construct: Call"), # ast.Call for eval
        ("a.b", {}, "Unsupported expression construct: Attribute"),
        ("'hello'", {}, "Unsupported constant type: <class 'str'>"), # String literal
        ("x = 5", {}, "Invalid syntax in expression"), # Assignment
        ("[1, 2, 3]", {}, "Unsupported expression construct: List"),
        ("{'a': 1}", {}, "Unsupported expression construct: Dict"),
        ("1 if True else 0", {}, "Unsupported expression construct: IfExp"),
    ]

    for expr, v_map, expected_err_part in test_cases_fail:
        try:
            res = evaluate_expression_safely(expr, v_map)
            print(f"FAIL (NoError): {expr} with {v_map} -> {res} (Expected error containing: '{expected_err_part}')")
        except ValueError as e:
            if expected_err_part.lower() in str(e).lower():
                print(f"PASS (Error): \"{expr}\" with {v_map} -> {e}")
            else:
                print(f"FAIL (WrongError): \"{expr}\" with {v_map} -> {e} (Expected error containing: '{expected_err_part}')")
        except Exception as e: # Should not happen if ValueError is comprehensive
            print(f"FAIL (OtherError): \"{expr}\" with {v_map} -> {type(e).__name__}: {e} (Expected ValueError containing: '{expected_err_part}')")

    # Test for non-string expression
    try:
        evaluate_expression_safely(123, {})
        print("FAIL (NoError): Non-string expression did not raise ValueError")
    except ValueError as e:
        if "expression must be a string" in str(e).lower():
            print(f"PASS (Error): Non-string expression -> {e}")
        else:
            print(f"FAIL (WrongError): Non-string expression -> {e}")

    # Test for non-dict variable_mapping
    try:
        evaluate_expression_safely("1+1", "not_a_dict")
        print("FAIL (NoError): Non-dict variable_mapping did not raise ValueError")
    except ValueError as e:
        if "variable mapping must be a dictionary" in str(e).lower():
            print(f"PASS (Error): Non-dict variable_mapping -> {e}")
        else:
            print(f"FAIL (WrongError): Non-dict variable_mapping -> {e}")
