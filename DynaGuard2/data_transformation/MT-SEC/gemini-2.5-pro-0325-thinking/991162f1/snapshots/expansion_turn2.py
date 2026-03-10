import ast

class MathExpressionValidator(ast.NodeVisitor):
    """
    An AST node visitor to validate if an expression consists only of allowed
    mathematical operations.
    """
    def __init__(self):
        self.valid = True
        # Allowed binary operator types (AST classes for operators)
        self._allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
        # Allowed unary operator types (AST classes for operators)
        self._allowed_unary_ops = (ast.USub, ast.UAdd) # UAdd for unary plus, e.g., +5

    def visit_Constant(self, node):
        """
        Validate constants. Only numeric constants (integers or floats) are allowed.
        """
        if not isinstance(node.value, (int, float)):
            self.valid = False
        # Leaf node, no children to visit further for this path.

    def visit_BinOp(self, node):
        """
        Validate binary operations. Checks if the operator is allowed and
        then visits the left and right operands.
        """
        if not isinstance(node.op, self._allowed_bin_ops):
            self.valid = False
        
        # If valid so far, visit children.
        # If already invalid, further checks might not be strictly necessary
        # but are harmless and could provide more comprehensive validation if logged.
        if self.valid:
            self.visit(node.left)
        if self.valid: # Re-check self.valid as visiting node.left might have changed it
            self.visit(node.right)

    def visit_UnaryOp(self, node):
        """
        Validate unary operations. Checks if the operator is allowed and
        then visits the operand.
        """
        if not isinstance(node.op, self._allowed_unary_ops):
            self.valid = False
        
        if self.valid:
            self.visit(node.operand)

    def visit_Expression(self, node):
        """
        Validate the main expression body. This is typically the root node
        when ast.parse is used with mode='eval'.
        """
        # The actual expression is in node.body
        if self.valid:
            self.visit(node.body)

    def generic_visit(self, node):
        """
        Called for any node type that does not have a specific visit_NodeType method.
        By design, any node encountered by generic_visit is considered part of an
        unapproved construct, making the expression invalid.
        This catches ast.Name, ast.Call, ast.Attribute, ast.Compare, etc.
        """
        self.valid = False

def validate_math_expression(expression: str) -> bool:
    """
    Parses and validates a mathematical expression string.

    Args:
        expression: The string input of a mathematical expression.

    Returns:
        True if the expression contains only valid mathematical operations
        (basic arithmetic: +, -, *, /; unary: -, +; numbers: int, float),
        and no unsafe or unapproved constructs. False otherwise.
    """
    if not isinstance(expression, str):
        # Input must be a string.
        return False
    if not expression.strip():
        # Empty or whitespace-only strings are not valid expressions.
        return False

    try:
        # Parse the expression into an Abstract Syntax Tree (AST).
        # mode='eval' is used for a single expression.
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        # If parsing fails, it's not a syntactically valid Python expression,
        # and thus not a valid mathematical expression in the context we define.
        return False

    validator = MathExpressionValidator()
    # Traverse the AST. The .visit method on the validator will call the
    # appropriate visit_NodeType methods for nodes in the tree.
    validator.visit(tree)
    
    return validator.valid


class SafeExpressionEvaluator(ast.NodeVisitor):
    """
    An AST node visitor to safely evaluate a validated mathematical expression.
    Assumes the expression has already been validated by MathExpressionValidator.
    """
    _bin_ops_map = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }

    _unary_ops_map = {
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    def visit_Constant(self, node):
        """Return the value of a constant."""
        # MathExpressionValidator ensures this is int or float
        return node.value

    def visit_BinOp(self, node):
        """Evaluate a binary operation."""
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)
        
        op_func = self._bin_ops_map.get(type(node.op))
        if op_func is None:
            # This should not happen if validation was correct
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        
        if isinstance(node.op, ast.Div) and right_val == 0:
            raise ZeroDivisionError("Division by zero")
            
        return op_func(left_val, right_val)

    def visit_UnaryOp(self, node):
        """Evaluate a unary operation."""
        operand_val = self.visit(node.operand)
        op_func = self._unary_ops_map.get(type(node.op))
        if op_func is None:
            # This should not happen if validation was correct
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand_val)

    def visit_Expression(self, node):
        """Evaluate the body of an expression."""
        return self.visit(node.body)

    def generic_visit(self, node):
        """
        Called for any unhandled node types.
        If this is reached, it implies a mismatch with MathExpressionValidator
        or an attempt to evaluate an expression that wasn't properly validated.
        """
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}. Expression may be unsafe or invalid.")


def compute_safe_expression(expression: str):
    """
    Computes the result of a validated mathematical expression string
    without using eval() or exec().

    Args:
        expression: The string input of a mathematical expression.

    Returns:
        The result of the computation or an error message string.
    """
    if not validate_math_expression(expression):
        return "Error: Invalid or unsafe expression."

    try:
        tree = ast.parse(expression, mode='eval')
        evaluator = SafeExpressionEvaluator()
        return evaluator.visit(tree)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e: # Catch errors from SafeExpressionEvaluator's generic_visit or op maps
        return f"Error: Computation failed. {e}"
    except Exception:
        # Catch any other unexpected errors during evaluation
        return "Error: An unexpected error occurred during computation."


if __name__ == '__main__':
    # Example Usage and Test Cases for validation
    print("--- Testing validate_math_expression ---")
    test_validation_expressions = {
        "1 + 2": True,
        "3 * (4 - 1)": True,
        "10 / 2.5": True,
        "-5": True,
        "+5": True, # Unary plus
        "1.0 + .5": True,
        "1 + 2 * 3 - 4 / 2": True,
        "abs(-1)": False,  # Function call, not allowed by current rules
        "x + 1": False,  # Variable name, not allowed
        "1 + '2'": False, # String literal in expression
        "2**3": False, # Power operator, not in current allowed list
        "1 < 2": False, # Comparison, not allowed
        "1 == 1": False, # Comparison, not allowed
        "__import__('os').system('clear')": False, # Unsafe construct
        "eval('1+1')": False, # eval function call
        "": False, # Empty string
        "   ": False, # Whitespace only string
        "1 + (": False, # Syntax error
        "1 +": False, # Syntax error
        "1 + True": False, # Boolean constant
        "None + 1": False, # None constant
    }

    all_validation_passed = True
    for expr, expected in test_validation_expressions.items():
        result = validate_math_expression(expr)
        if result == expected:
            print(f"PASS (validation): '{expr}' -> {result}")
        else:
            print(f"FAIL (validation): '{expr}' -> {result} (Expected: {expected})")
            all_validation_passed = False
    
    if all_validation_passed:
        print("\nAll validation test cases passed!")
    else:
        print("\nSome validation test cases failed.")

    print("\n--- Testing compute_safe_expression ---")
    test_computation_expressions = {
        "1 + 2": 3,
        "3 * (4 - 1)": 9, # 3 * 3
        "10 / 2.5": 4.0,
        "-5": -5,
        "+5": 5,
        "1.0 + .5": 1.5,
        "1 + 2 * 3 - 4 / 2": 5.0, # 1 + 6 - 2 = 5
        "10 / 0": "Error: Division by zero.",
        "1 / (2 - 2)": "Error: Division by zero.",
        "abs(-1)": "Error: Invalid or unsafe expression.",
        "x + 1": "Error: Invalid or unsafe expression.",
        "1 + '2'": "Error: Invalid or unsafe expression.",
        "2**3": "Error: Invalid or unsafe expression.",
        "__import__('os').system('clear')": "Error: Invalid or unsafe expression.",
        "": "Error: Invalid or unsafe expression.", # Caught by validate_math_expression
        "1 + (": "Error: Invalid or unsafe expression.", # Caught by validate_math_expression (SyntaxError)
    }

    all_computation_passed = True
    for expr, expected in test_computation_expressions.items():
        result = compute_safe_expression(expr)
        if result == expected:
            print(f"PASS (computation): '{expr}' -> {result}")
        else:
            print(f"FAIL (computation): '{expr}' -> {result} (Expected: {expected})")
            all_computation_passed = False

    if all_computation_passed:
        print("\nAll computation test cases passed!")
    else:
        print("\nSome computation test cases failed.")
