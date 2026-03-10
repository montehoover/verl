import re
import operator
import ast
import logging

# Initialize logger
logger = logging.getLogger(__name__)

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
    logger.debug(f"Evaluating node: {type(node).__name__}, AST: {ast.dump(node) if isinstance(node, ast.AST) else node}")
    if isinstance(node, ast.Constant):
        # In Python 3.8+, strings and bytes are also ast.Constant.
        # We only want to process numbers here.
        if not isinstance(node.value, (int, float)):
            logger.error(f"Unsupported constant type: {type(node.value)} for value {node.value}")
            raise ValueError(f"Unsupported constant type: {type(node.value)}. Only numeric constants are allowed.")
        logger.debug(f"Constant node: {node.value}")
        return node.value
    elif isinstance(node, ast.Name):
        if node.id not in variable_mapping:
            logger.error(f"Undefined variable: {node.id}")
            raise ValueError(f"Undefined variable: {node.id}")
        value = variable_mapping[node.id]
        # Ensure variables resolve to numbers
        if not isinstance(value, (int, float)):
            logger.error(f"Variable '{node.id}' must resolve to a number, got {type(value)} with value {value}")
            raise ValueError(f"Variable '{node.id}' must resolve to a number, got {type(value)}")
        logger.debug(f"Name node: '{node.id}', value: {value}")
        return value
    elif isinstance(node, ast.BinOp):
        logger.debug(f"Evaluating BinOp: {type(node.op).__name__}")
        left_val = _recursive_eval_ast_node(node.left, variable_mapping)
        right_val = _recursive_eval_ast_node(node.right, variable_mapping)
        op_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if op_func is None:
            logger.error(f"Unsupported binary operator: {type(node.op).__name__}")
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        try:
            result = op_func(left_val, right_val)
            logger.debug(f"BinOp '{type(node.op).__name__}' on ({left_val}, {right_val}) -> {result}")
            return result
        except ZeroDivisionError:
            logger.error("Division by zero encountered in BinOp")
            raise ValueError("Division by zero")
    elif isinstance(node, ast.UnaryOp):
        logger.debug(f"Evaluating UnaryOp: {type(node.op).__name__}")
        operand_val = _recursive_eval_ast_node(node.operand, variable_mapping)
        op_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if op_func is None:
            logger.error(f"Unsupported unary operator: {type(node.op).__name__}")
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        result = op_func(operand_val)
        logger.debug(f"UnaryOp '{type(node.op).__name__}' on ({operand_val}) -> {result}")
        return result
    else:
        # Catch any other node types (like ast.Call, ast.Attribute, ast.Subscript, etc.)
        logger.error(f"Unsupported expression construct: {type(node).__name__}")
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

def _parse_to_ast_body(math_expression: str) -> ast.AST:
    """
    Parses a mathematical expression string into an AST body.
    Raises ValueError for syntax errors or invalid AST structure.
    """
    logger.debug(f"Attempting to parse expression to AST: '{math_expression}'")
    try:
        # Parse the expression in 'eval' mode, which expects a single expression
        parsed_ast = ast.parse(math_expression, mode='eval')

        # The body of an 'eval' mode AST is an ast.Expression node,
        # its 'body' attribute contains the actual expression node.
        if not isinstance(parsed_ast, ast.Expression) or not hasattr(parsed_ast, 'body'):
            # This case should ideally not be reached if ast.parse(mode='eval') works as expected
            # and the expression is valid.
            logger.error(f"Invalid AST structure after parsing '{math_expression}'. Parsed as: {type(parsed_ast)}")
            raise ValueError("Invalid expression structure after parsing.")
        logger.debug(f"Successfully parsed '{math_expression}'. AST body: {ast.dump(parsed_ast.body)}")
        return parsed_ast.body
    except SyntaxError as e:
        logger.error(f"Syntax error parsing expression: '{math_expression}'. Error: {e}", exc_info=True)
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
    logger.info(f"Evaluating expression: '{math_expression}' with variables: {variable_mapping}")

    if not isinstance(math_expression, str):
        logger.error("Expression must be a string.", exc_info=True)
        raise ValueError("Expression must be a string.")
    if not isinstance(variable_mapping, dict):
        logger.error("Variable mapping must be a dictionary.", exc_info=True)
        raise ValueError("Variable mapping must be a dictionary.")

    try:
        # Validate variable names and their values in the mapping
        for var_name, var_value in variable_mapping.items():
            if not isinstance(var_name, str) or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                logger.error(f"Invalid variable name: '{var_name}' in mapping: {variable_mapping}", exc_info=True)
                raise ValueError(f"Invalid variable name: '{var_name}'")
            if not isinstance(var_value, (int, float)):
                logger.error(f"Variable '{var_name}' must be a number, got {type(var_value).__name__} in mapping: {variable_mapping}", exc_info=True)
                raise ValueError(f"Variable '{var_name}' must be a number, got {type(var_value).__name__}")
        logger.debug("Variable mapping validated successfully.")

        # Pipeline Step 1: Parse the expression string to an AST body.
        ast_body = _parse_to_ast_body(math_expression)

        # Pipeline Step 2: Evaluate the AST body using the variable mapping.
        result = _recursive_eval_ast_node(ast_body, variable_mapping)
        
        # Ensure the result from evaluation is a number before converting to string
        if not isinstance(result, (int, float)):
            logger.error(f"Evaluation result is not a number: {result} (type: {type(result)})", exc_info=True)
            raise ValueError("Evaluation did not result in a number.")
        
        result_str = str(result)
        logger.info(f"Successfully evaluated expression '{math_expression}'. Result: {result_str}")
        return result_str

    except ValueError as e: # Catches ValueErrors from validation, parsing, or evaluation
        logger.error(f"ValueError during evaluation of '{math_expression}': {e}", exc_info=True)
        raise
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during evaluation of '{math_expression}': {e}", exc_info=True)
        raise ValueError(f"Failed to evaluate expression due to an unexpected error: {type(e).__name__} - {e}")

if __name__ == '__main__':
    # Configure basic logging for console output
    logging.basicConfig(
        level=logging.DEBUG, # Set to logging.INFO for less verbose output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example Usage:
    logger.info("Starting expression evaluation examples...")
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
        logger.error(f"Error in basic examples: {e}", exc_info=True)
        print(f"Error in basic examples: {e}")

    logger.info("Starting error case testing...")
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
                # Log the unexpected error message for easier debugging of tests
                logger.warning(f"FAIL (WrongError) for \"{expr}\": Expected '{expected_err_part}', Got '{str(e)}'")
                print(f"FAIL (WrongError): \"{expr}\" with {v_map} -> {e} (Expected error containing: '{expected_err_part}')")
        except Exception as e: # Should not happen if ValueError is comprehensive
            logger.error(f"FAIL (OtherError) for \"{expr}\": Unexpected {type(e).__name__}: {e}", exc_info=True)
            print(f"FAIL (OtherError): \"{expr}\" with {v_map} -> {type(e).__name__}: {e} (Expected ValueError containing: '{expected_err_part}')")

    # Test for non-string expression
    logger.info("Testing non-string expression input.")
    try:
        evaluate_expression_safely(123, {})
        print("FAIL (NoError): Non-string expression did not raise ValueError")
    except ValueError as e:
        if "expression must be a string" in str(e).lower():
            print(f"PASS (Error): Non-string expression -> {e}")
        else:
            print(f"FAIL (WrongError): Non-string expression -> {e}")

    # Test for non-dict variable_mapping
    logger.info("Testing non-dict variable_mapping input.")
    try:
        evaluate_expression_safely("1+1", "not_a_dict")
        print("FAIL (NoError): Non-dict variable_mapping did not raise ValueError")
    except ValueError as e:
        if "variable mapping must be a dictionary" in str(e).lower():
            print(f"PASS (Error): Non-dict variable_mapping -> {e}")
        else:
            print(f"FAIL (WrongError): Non-dict variable_mapping -> {e}")
