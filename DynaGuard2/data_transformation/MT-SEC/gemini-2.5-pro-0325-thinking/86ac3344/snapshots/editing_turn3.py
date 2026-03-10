import re
import ast
from typing import List, Union, Any

def evaluate_expression(input_str: str) -> List[Union[int, float]]:
    """
    Identifies and returns a list of all numbers present in input_str.

    Numbers can be integers or floating-point values.
    The input string is expected to contain numbers separated by whitespace.

    Args:
        input_str: The string to parse.

    Returns:
        A list of numbers (int or float) found in the string.

    Raises:
        ValueError: If input_str contains any characters other than digits, '.', '-',
                    or whitespace (e.g., alphabetic characters or special symbols).
                    Also raises ValueError if a token (part of the string separated
                    by whitespace) forms a malformed number (e.g., "1.2.3", "--").
    """
    # First, check for prohibited characters in the entire string.
    # Allowed characters are digits, decimal points, hyphens, and whitespace.
    # This regex pattern matches strings that ONLY contain these allowed characters.
    allowed_chars_pattern = r"^[0-9.\-\s]*$"
    if not re.fullmatch(allowed_chars_pattern, input_str):
        # Find the first prohibited character for a more specific error message.
        first_prohibited_char = "" # Will be set if pattern failed for non-empty string
        for char_in_str in input_str:
            # A character is prohibited if it's not a digit, not '.', not '-', and not whitespace.
            if not (char_in_str.isdigit() or char_in_str == '.' or char_in_str == '-' or char_in_str.isspace()):
                first_prohibited_char = char_in_str
                break
        # If re.fullmatch failed on a non-empty string, first_prohibited_char must have been found.
        # If input_str is empty, re.fullmatch(allowed_chars_pattern, "") passes, so this code path is not taken.
        raise ValueError(f"Input string contains a prohibited character: '{first_prohibited_char}'")

    # Regex for a valid number (integer or float, handles positive/negative, .5, 5.)
    # Does not support scientific notation like 1e5.
    number_pattern_strict = r"-?(?:\d+(?:\.\d*)?|\.\d+)"

    tokens = input_str.split()
    
    # If input_str is empty or only whitespace, tokens will be an empty list,
    # and an empty list will be returned, which is correct.

    numbers_found = []
    for token in tokens:
        if re.fullmatch(number_pattern_strict, token):
            # Convert to int if it doesn't have a decimal point, otherwise float.
            # This preserves integer types where appropriate.
            if '.' in token:
                num = float(token)
            else:
                num = int(token)
            numbers_found.append(num)
        else:
            # After the initial character check, tokens will only contain digits, '.', or '-'.
            # If such a token does not match the number_pattern_strict,
            # it's a malformed number or an invalid sequence of allowed characters (e.g., "--", "1.2.3", ".").
            raise ValueError(f"Invalid input: '{token}' is a malformed number or an invalid sequence of allowed characters.")
            
    return numbers_found


# --- New function secure_eval_expression and helper class ---

# Allowed AST node types
ALLOWED_NODES = {
    ast.Expression,  # The overall expression
    ast.Constant,    # For Python 3.8+ (numbers, None, True, False)
    ast.Num,         # For numbers (deprecated in 3.8, but good for compatibility)
    ast.BinOp,       # Binary operations like +, -, *, /
    ast.UnaryOp,     # Unary operations like - (negation)
    ast.Add,         # Addition operator
    ast.Sub,         # Subtraction operator
    ast.Mult,        # Multiplication operator
    ast.Div,         # Division operator
    ast.UAdd,        # Unary plus
    ast.USub,        # Unary minus (negation)
}

# Allowed operator types within BinOp and UnaryOp
ALLOWED_OPERATORS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
}

class SafeExpressionEvaluator(ast.NodeVisitor):
    """
    Safely evaluates an AST expression node.
    Only allows basic arithmetic operations and numbers.
    """
    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node: ast.Num) -> Union[int, float]: # For Python < 3.8
        if isinstance(node.n, (int, float)):
            return node.n
        raise ValueError(f"Unsupported number type: {type(node.n)}")

    def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        # Should not be reached if ALLOWED_OPERATORS is checked correctly
        raise ValueError(f"Unhandled binary operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        # Should not be reached if ALLOWED_OPERATORS is checked correctly
        raise ValueError(f"Unhandled unary operator: {type(node.op).__name__}")

    def generic_visit(self, node: ast.AST) -> None:
        """
        Called if no explicit visitor function exists for a node type.
        This ensures only whitelisted nodes are processed.
        """
        # Check if the node itself is an allowed type (e.g. Expression)
        # or if it's an operator type (which are handled by their parent BinOp/UnaryOp)
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        # For nodes like ast.Expression, we need to continue visiting children.
        # For operator nodes (Add, Sub, etc.), they are not visited directly but used by BinOp/UnaryOp.
        super().generic_visit(node)


def secure_eval_expression(input_str: str) -> Union[int, float]:
    """
    Safely evaluates a mathematical expression string using AST.

    Only allows basic arithmetic operations (+, -, *, /), numbers, and parentheses.
    Raises ValueError for any unsafe characters, syntax errors, or restricted operations.

    Args:
        input_str: The mathematical expression string to evaluate.

    Returns:
        The numerical result of the evaluation.

    Raises:
        ValueError: If the input string contains unsafe characters,
                    is syntactically incorrect, or uses disallowed operations/elements.
    """
    # Basic character validation: allow digits, operators, parentheses, dot, and whitespace.
    # This is a pre-filter; ast.parse will do the thorough syntax check.
    allowed_char_pattern = r"^[0-9\s\.\+\-\*\/\(\)]*$"
    if not re.fullmatch(allowed_char_pattern, input_str):
        # Find the first prohibited character for a more specific error message.
        first_prohibited_char = ""
        for char_in_str in input_str:
            if not (char_in_str.isdigit() or char_in_str.isspace() or \
                    char_in_str in ['.', '+', '-', '*', '/', '(', ')']):
                first_prohibited_char = char_in_str
                break
        raise ValueError(f"Input string contains prohibited character: '{first_prohibited_char}'")

    if not input_str.strip():
        raise ValueError("Input string cannot be empty or only whitespace.")

    try:
        # Parse the expression in 'eval' mode, which expects a single expression.
        # The `ast.parse` function itself can raise SyntaxError for malformed expressions.
        tree = ast.parse(input_str, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except Exception as e: # Catch other potential parsing errors
        raise ValueError(f"Error parsing expression: {e}")

    # Validate all nodes in the AST tree before evaluation
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    evaluator = SafeExpressionEvaluator()
    try:
        # The visit method on the root of an 'eval' mode AST (Expression node)
        # will trigger the chain of visits.
        result = evaluator.visit(tree.body) # tree.body is the actual expression node
        if not isinstance(result, (int, float)):
            # This case should ideally be caught by the visitor's type checks
            raise ValueError("Evaluation did not result in a number.")
        return result
    except ZeroDivisionError: # Explicitly catch and re-raise as ValueError
        raise ValueError("Division by zero")
    # ValueErrors raised by the visitor will propagate up.
    # Other runtime errors during evaluation should be caught if necessary,
    # but the strict AST checking should prevent most.
