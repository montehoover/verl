import re
import ast

def validate_math_expression(expression: str) -> bool:
    """
    Validates a mathematical expression string.

    Args:
        expression: The string to validate.

    Returns:
        True if the string only contains numbers, arithmetic operators
        (+, -, *, /), and spaces, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    return bool(re.fullmatch(allowed_chars_pattern, expression))

def compute(operand1: float, operator: str, operand2: float) -> float:
    """
    Computes the result of a basic arithmetic operation.

    Args:
        operand1: The first operand (left side).
        operator: The operator string ('+', '-', '*', '/').
        operand2: The second operand (right side).

    Returns:
        The result of the operation.

    Raises:
        ValueError: If the operator is unknown.
        ZeroDivisionError: If division by zero is attempted.
    """
    if operator == '+':
        return operand1 + operand2
    elif operator == '-':
        return operand1 - operand2
    elif operator == '*':
        return operand1 * operand2
    elif operator == '/':
        if operand2 == 0:
            raise ZeroDivisionError("Division by zero")
        return operand1 / operand2
    else:
        raise ValueError(f"Unknown operator: {operator}")

def parse_expression(expression: str) -> float:
    """
    Parses and evaluates a validated mathematical expression string.
    Respects standard operator precedence (*, / before +, -).
    Assumes the expression string has already been validated for allowed characters.

    Args:
        expression: The mathematical expression string to evaluate.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If the expression is malformed, contains invalid characters,
                    is empty, or results in an invalid state.
        ZeroDivisionError: If the expression attempts division by zero.
    """
    if not validate_math_expression(expression):
        raise ValueError("Expression contains invalid characters.")

    # Tokenize numbers (including negative floats) and operators.
    # Removes all spaces first, then tokenizes.
    # Example: "3 + -5 * 2" -> "3+-5*2" -> ['3', '+', '-5', '*', '2']
    tokens = re.findall(r"-?\d+\.?\d*|[+\-*/]", expression.replace(" ", ""))

    if not tokens:
        raise ValueError("Expression is empty or contains only spaces.")

    values_stack = []
    ops_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    def _apply_top_op():
        try:
            op = ops_stack.pop()
            right_operand = values_stack.pop()
            left_operand = values_stack.pop()
        except IndexError:
            # This error means not enough operands for an operator,
            # e.g. "5 * " or "+ 5" or "5 6" (if 6 was an op)
            raise ValueError("Malformed expression: insufficient operands for an operator or invalid structure.")
        
        result = compute(left_operand, op, right_operand) # compute can raise ZeroDivisionError
        values_stack.append(result)

    for token in tokens:
        try:
            # If token is a number
            num_value = float(token)
            values_stack.append(num_value)
        except ValueError:
            # Token is an operator
            if token not in precedence:
                # This should ideally not be reached if validate_math_expression is comprehensive
                # and tokenization is correct.
                raise ValueError(f"Unknown token or operator: {token}")

            # While ops_stack is not empty, its top is an operator,
            # and its precedence is >= current token's precedence
            while (ops_stack and
                   ops_stack[-1] in precedence and
                   precedence[ops_stack[-1]] >= precedence[token]):
                _apply_top_op()
            ops_stack.append(token)

    # After all tokens are processed, apply any remaining operations in the stack
    while ops_stack:
        _apply_top_op()

    if len(values_stack) == 1 and not ops_stack:
        return values_stack[0]
    else:
        # This can happen for malformed expressions like "3 4" (parsed as two numbers without operator)
        # or if the expression was incomplete or structurally unsound in a way not caught earlier.
        raise ValueError("Malformed expression: final state is invalid (e.g. multiple values left, or operators remaining).")

def secure_eval_expression(expression: str) -> float:
    """
    Securely validates and evaluates a mathematical expression string.

    Args:
        expression: The mathematical expression string to evaluate.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If the input expression contains invalid characters,
                    is malformed, or attempts unsafe operations.
        ZeroDivisionError: If the expression attempts division by zero.
    """
    if not validate_math_expression(expression):
        raise ValueError("Expression contains invalid characters and cannot be evaluated.")
    
    # parse_expression will raise ValueError for malformed expressions or ZeroDivisionError
    return parse_expression(expression)
