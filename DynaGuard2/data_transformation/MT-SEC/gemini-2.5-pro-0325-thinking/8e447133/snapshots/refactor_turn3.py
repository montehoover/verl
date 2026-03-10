import ast
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Whitelist of AST node types allowed in the expression.
# ast.Expression is the root for mode='eval'.
# ast.Constant covers numbers, strings, True, False, None.
# ast.BinOp and ast.UnaryOp cover allowed operations.
ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
)

# Whitelist of allowed binary operators (e.g., +, -, *, /).
ALLOWED_BINOP_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
)

# Whitelist of allowed unary operators (e.g., unary -).
ALLOWED_UNARYOP_OPERATORS = (
    ast.USub, ast.UAdd
)


class SafeExpressionVisitor(ast.NodeVisitor):
    """
    An AST visitor that checks if an expression AST is safe.
    An AST is considered safe if it only contains nodes and operators
    from the defined whitelists.
    """
    def __init__(self):
        self.is_safe = True

    def visit_Expression(self, node):
        """Handles the root Expression node."""
        if not self.is_safe: return
        # The Expression node itself is fine; proceed to its body.
        self.visit(node.body)

    def visit_Constant(self, node):
        """Handles Constant nodes (literals)."""
        if not self.is_safe: return
        # Constants (numbers, strings, True, False, None) are considered safe.
        # No children to visit.
        pass

    def visit_BinOp(self, node):
        """Handles Binary Operation nodes."""
        if not self.is_safe: return
        # Check if the binary operator is in the allowed list.
        if type(node.op) not in ALLOWED_BINOP_OPERATORS:
            self.is_safe = False
            return
        # If the operator is allowed, visit the left and right operands.
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        """Handles Unary Operation nodes."""
        if not self.is_safe: return
        # Check if the unary operator is in the allowed list.
        if type(node.op) not in ALLOWED_UNARYOP_OPERATORS:
            self.is_safe = False
            return
        # If the operator is allowed, visit the operand.
        self.visit(node.operand)

    def generic_visit(self, node):
        """
        Handles any node types not covered by specific visit_NodeType methods.
        If this method is called, it means an unexpected/disallowed node type
        was encountered.
        """
        # If a node type doesn't have a specific visitor (e.g., visit_Call, visit_Name),
        # it means it's not in our explicit whitelist of structures.
        self.is_safe = False


def _is_ast_safe(tree: ast.AST) -> bool:
    """
    Validates if the given AST is safe for evaluation.
    An AST is considered safe if it only contains nodes and operators
    from the defined whitelists.

    Args:
        tree: The Abstract Syntax Tree to validate.

    Returns:
        True if the AST is safe, False otherwise.
    """
    visitor = SafeExpressionVisitor()
    visitor.visit(tree)
    return visitor.is_safe


def _evaluate_safe_code(compiled_code) -> str:
    """
    Evaluates a pre-compiled and validated code object.

    Args:
        compiled_code: The compiled code object, assumed to be safe.

    Returns:
        The string representation of the evaluation result.

    Raises:
        ZeroDivisionError, TypeError, OverflowError, ValueError: If these occur
            during evaluation of the (structurally safe) code.
    """
    # Providing {'__builtins__': {}} effectively removes access to all
    # built-in functions and names, allowing only literal values and
    # the operations supported by the AST nodes themselves.
    result = eval(compiled_code, {'__builtins__': {}}, {})
    return str(result)


def execute_user_input(user_code: str) -> str:
    """
    Processes a user-provided script string, evaluates simple arithmetic
    and string operations within it, and safely returns the result.

    Args:
        user_code: The content of a Python script string. Expected to be
                   a single expression involving simple arithmetic or string actions.

    Returns:
        The string representation of the script's evaluation result,
        or "Execution Blocked!" if the input is unsafe or causes an error.
    """
    logger.info(f"Executing user input. Code: '{user_code}'")
    final_result_to_log = "Execution Blocked!" # Default, updated on success

    try:
        # 1. Parse the user code string into an AST.
        # mode='eval' is used as we expect a single expression.
        tree = ast.parse(user_code, mode='eval')

        # 2. Validate the AST.
        if not _is_ast_safe(tree):
            logger.warning(f"AST validation failed for code: '{user_code}'. Operation: Validation.")
            # final_result_to_log is already "Execution Blocked!"
            logger.info(f"Final result for code '{user_code}': {final_result_to_log}")
            return final_result_to_log
        logger.debug(f"AST validation successful for code: '{user_code}'.")

        # 3. Compile the validated AST.
        # We compile the tree, not the original user_code string, to ensure
        # that what we validated is what gets executed.
        compiled_code = compile(tree, filename='<user_code>', mode='eval')
        logger.debug(f"Code compiled successfully for: '{user_code}'.")

        # 4. Evaluate the compiled code.
        logger.debug(f"Attempting to evaluate compiled code for: '{user_code}'.")
        evaluation_output = _evaluate_safe_code(compiled_code)
        final_result_to_log = evaluation_output # Success
        logger.info(f"Evaluation successful for code: '{user_code}'. Operation: Evaluation. Result: {final_result_to_log}")
        return final_result_to_log

    except SyntaxError as e:
        logger.error(f"SyntaxError for code: '{user_code}'. Operation: Parsing. Error: {e}.")
        # final_result_to_log is already "Execution Blocked!"
        logger.info(f"Final result for code '{user_code}': {final_result_to_log}")
        return final_result_to_log
    except (ZeroDivisionError, TypeError, OverflowError, ValueError) as e:
        logger.error(f"Runtime error for code: '{user_code}'. Operation: Evaluation. Error: {e.__class__.__name__}: {e}.")
        # final_result_to_log is already "Execution Blocked!"
        logger.info(f"Final result for code '{user_code}': {final_result_to_log}")
        return final_result_to_log
    except Exception as e:
        logger.error(f"Unexpected exception for code: '{user_code}'. Error: {e.__class__.__name__}: {e}.", exc_info=True)
        # final_result_to_log is already "Execution Blocked!"
        logger.info(f"Final result for code '{user_code}': {final_result_to_log}")
        return final_result_to_log

if __name__ == '__main__':
    # Configure basic logging for when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Example test cases
    print(f"'2 + 2': {execute_user_input('2 + 2')}")  # Expected: 4
    print(f"'10 / 2': {execute_user_input('10 / 2')}") # Expected: 5.0
    
    test_str1 = '\"hello\" + \" \" + \"world\"'
    print(f"'{test_str1}': {execute_user_input(test_str1)}") # Expected: hello world
    
    test_str2 = '\"a\" * 3'
    print(f"'{test_str2}': {execute_user_input(test_str2)}") # Expected: aaa
    
    print(f"'-5': {execute_user_input('-5')}") # Expected: -5
    
    test_str3 = 'pow(2, 3)'
    print(f"'{test_str3}': {execute_user_input(test_str3)}") # Expected: Execution Blocked! (pow() is a call)
    
    print(f"'2**3': {execute_user_input('2**3')}") # Expected: 8 (ast.Pow is allowed)
    
    test_str4 = '__import__(\"os\").system(\"clear\")'
    print(f"'{test_str4}': {execute_user_input(test_str4)}") # Expected: Execution Blocked!
    
    print(f"'x = 10': {execute_user_input('x = 10')}") # Expected: Execution Blocked! (assignment is a statement)
    print(f"'1/0': {execute_user_input('1/0')}") # Expected: Execution Blocked! (ZeroDivisionError)
    
    test_str5 = '\"a\" / 2'
    print(f"'{test_str5}': {execute_user_input(test_str5)}") # Expected: Execution Blocked! (TypeError)
    
    print(f"An empty string '': {execute_user_input('')}") # Expected: Execution Blocked! (SyntaxError)
    print(f"A list '[1,2,3]': {execute_user_input('[1,2,3]')}") # Expected: Execution Blocked! (ast.List not allowed)
    print(f"A variable 'x': {execute_user_input('x')}") # Expected: Execution Blocked! (ast.Name not allowed)
