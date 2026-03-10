import ast

class SafeExpressionEvaluator:
    """
    Safely evaluates arithmetic and logical expressions using AST validation.
    """
    ALLOWED_NODE_TYPES = {
        ast.Expression,  # Root node for 'eval' mode
        # Literals/Constants:
        ast.Constant,    # Python 3.8+ for numbers, strings (filtered), bools, None (filtered)
        ast.Num,         # Python < 3.8 for numbers
        ast.NameConstant,# Python < 3.8 for True, False, None (None filtered)
        # Variables/Names:
        ast.Name,        # For 'True', 'False'
        ast.Load,        # Context for loading a variable
        # Operators:
        ast.BinOp,       # e.g., +, -, *, /
        ast.UnaryOp,     # e.g., -, not
        ast.BoolOp,      # e.g., and, or
        ast.Compare,     # e.g., >, <, ==
        # Specific operator types (visited by ast.walk):
        ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.And, ast.Or,
        ast.Not, ast.USub, ast.UAdd,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    }
    ALLOWED_NAMES = {'True', 'False'}

    def is_safe(self, expression_str: str) -> bool:
        """
        Checks if the expression string is safe to evaluate by parsing it into an
        Abstract Syntax Tree (AST) and validating all nodes.
        """
        try:
            tree = ast.parse(expression_str, mode='eval')
        except SyntaxError:
            return False # Not valid Python syntax

        for node in ast.walk(tree):
            if type(node) not in self.ALLOWED_NODE_TYPES:
                return False # Disallowed node type

            if isinstance(node, ast.Name):
                if node.id not in self.ALLOWED_NAMES:
                    return False # Disallowed variable name

            # Python 3.8+ uses ast.Constant for literals
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, bool)):
                    # Disallow strings, bytes, None, complex numbers, etc.
                    return False
            
            # Python < 3.8 uses ast.NameConstant for True, False, None
            elif hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
                if node.value not in {True, False}:
                    # Disallow None from NameConstant
                    return False
            # ast.Num (numbers for Python < 3.8) is implicitly allowed if in ALLOWED_NODE_TYPES.
        return True

    def evaluate(self, expression_str: str):
        """
        Evaluates the expression string if it's deemed safe.
        Returns the result or an error message.
        """
        if not isinstance(expression_str, str) or not self.is_safe(expression_str):
            return 'Invalid Expression!'
        try:
            # Using a restricted globals and locals dictionary for eval.
            # __builtins__ is emptied to prevent access to built-in functions.
            # True and False are keywords and will work without being in locals.
            result = eval(expression_str, {"__builtins__": {}}, {})
            return result
        except (SyntaxError, NameError, TypeError, ZeroDivisionError, Exception):
            # Catch any other runtime errors during evaluation
            return 'Invalid Expression!'

_EVALUATOR_INSTANCE = SafeExpressionEvaluator()

def evaluate_math_expression(expression: str):
    """
    Evaluates a string containing a simple arithmetic or logical expression.

    Args:
        expression: A string representing the arithmetic or logical expression.
                    Supports +, -, *, /, and, or, not, True, False, and comparisons.

    Returns:
        The calculated result if the expression is valid and safe,
        otherwise an error message 'Invalid Expression!'.
    """
    return _EVALUATOR_INSTANCE.evaluate(expression)

def calculate_user_input(input_expr: str):
    """
    Evaluates a user-provided expression string containing basic arithmetic or logical operations.

    Args:
        input_expr: A string representing the arithmetic or logical expression.

    Returns:
        The calculated result if the expression is valid and safe,
        otherwise 'Processing Failed!'.
    """
    result = _EVALUATOR_INSTANCE.evaluate(input_expr)
    if result == 'Invalid Expression!':
        return 'Processing Failed!'
    return result

if __name__ == '__main__':
    # Test cases for evaluate_math_expression (existing function)
    print("--- Testing evaluate_math_expression ---")
    # Test cases for arithmetic operations
    print(f"Expression: '3 + 5', Result: {evaluate_math_expression('3 + 5')}")
    print(f"Expression: '10 - 2', Result: {evaluate_math_expression('10 - 2')}")
    print(f"Expression: '4 * 6', Result: {evaluate_math_expression('4 * 6')}")
    print(f"Expression: '8 / 2', Result: {evaluate_math_expression('8 / 2')}")
    print(f"Expression: '5 / 0', Result: {evaluate_math_expression('5 / 0')}") # ZeroDivisionError
    print(f"Expression: '3 + 5 * 2', Result: {evaluate_math_expression('3 + 5 * 2')}")
    print(f"Expression: '(3 + 5) * 2', Result: {evaluate_math_expression('(3 + 5) * 2')}")
    
    # Test cases for logical operations
    print(f"Expression: 'True and False', Result: {evaluate_math_expression('True and False')}")
    print(f"Expression: 'True or False', Result: {evaluate_math_expression('True or False')}")
    print(f"Expression: 'not True', Result: {evaluate_math_expression('not True')}")
    print(f"Expression: '(True and False) or (not False)', Result: {evaluate_math_expression('(True and False) or (not False)')}")
    
    # Test cases for combined arithmetic and logical (via comparisons)
    print(f"Expression: '3 > 2 and 1 < 0', Result: {evaluate_math_expression('3 > 2 and 1 < 0')}")
    print(f"Expression: '10 == 10 or 5 != 5', Result: {evaluate_math_expression('10 == 10 or 5 != 5')}")
    print(f"Expression: 'not (5 < 2)', Result: {evaluate_math_expression('not (5 < 2)')}")

    # Test cases for invalid expressions
    print(f"Expression: '3 + ', Result: {evaluate_math_expression('3 + ')}") # SyntaxError
    print(f"Expression: 'abc + 5', Result: {evaluate_math_expression('abc + 5')}") # Disallowed Name
    print(f"Expression: '10 / (2 - 2)', Result: {evaluate_math_expression('10 / (2 - 2)')}") # ZeroDivisionError
    print(f"Expression: 'import os', Result: {evaluate_math_expression('import os')}") # Disallowed AST node (Import)
    expr_import_os = '__import__("os").getcwd()'
    print(f"Expression: '{expr_import_os}', Result: {evaluate_math_expression(expr_import_os)}") # Disallowed
    expr_eval_1_plus_1 = 'eval("1+1")'
    print(f"Expression: '{expr_eval_1_plus_1}', Result: {evaluate_math_expression(expr_eval_1_plus_1)}") # Disallowed Name 'eval'
    expr_list_comp = '[x for x in [1,2]]'
    print(f"Expression: '{expr_list_comp}', Result: {evaluate_math_expression(expr_list_comp)}") # Disallowed ListComp
    expr_1_plus_str_2 = '1 + "2"'
    print(f"Expression: '{expr_1_plus_str_2}', Result: {evaluate_math_expression(expr_1_plus_str_2)}") # TypeError (but caught as Invalid by AST check for string literal)
    expr_dict_a_1 = '{"a": 1}'
    print(f"Expression: '{expr_dict_a_1}', Result: {evaluate_math_expression(expr_dict_a_1)}") # Disallowed Dict

    # Test cases for calculate_user_input (new function)
    print("\n--- Testing calculate_user_input ---")
    print(f"Input: '10 + 20', Output: {calculate_user_input('10 + 20')}")
    print(f"Input: 'True and False', Output: {calculate_user_input('True and False')}")
    print(f"Input: '100 / 2', Output: {calculate_user_input('100 / 2')}")
    print(f"Input: '100 / 0', Output: {calculate_user_input('100 / 0')}") # Expected: Processing Failed!
    print(f"Input: '5 > 3 or 1 < 0', Output: {calculate_user_input('5 > 3 or 1 < 0')}")
    print(f"Input: 'not (10 == 20)', Output: {calculate_user_input('not (10 == 20)')}")
    print(f"Input: 'some_var + 10', Output: {calculate_user_input('some_var + 10')}") # Expected: Processing Failed!
    print(f"Input: 'import sys', Output: {calculate_user_input('import sys')}") # Expected: Processing Failed!
    expr_malicious_input = '__import__("os").system("echo unsafe")'
    print(f"Input: '{expr_malicious_input}', Output: {calculate_user_input(expr_malicious_input)}") # Expected: Processing Failed!
    print(f"Input: '(10 + 5) * 2 / (4 - 2)', Output: {calculate_user_input('(10 + 5) * 2 / (4 - 2)')}")
    print(f"Input: '3 +', Output: {calculate_user_input('3 +')}") # Expected: Processing Failed! (Syntax Error)
