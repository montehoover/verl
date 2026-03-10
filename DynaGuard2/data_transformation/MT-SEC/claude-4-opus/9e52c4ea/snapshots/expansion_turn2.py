import re
import ast
import operator

def validate_math_expression(expression):
    """
    Validates a mathematical expression to ensure it only contains
    numbers and basic arithmetic operators (+, -, *, /, %, //, **).
    Also checks for balanced parentheses.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Remove all whitespace for easier processing
    expression = expression.replace(" ", "")
    
    # Check if expression is empty
    if not expression:
        return False
    
    # Define allowed characters: digits, operators, parentheses, and decimal points
    allowed_pattern = r'^[0-9+\-*/%(). ]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False
    
    if paren_count != 0:
        return False
    
    # Check for invalid patterns
    invalid_patterns = [
        r'[+\-*/]{2,}',  # Multiple operators in a row (except --)
        r'^[*/]',        # Expression starting with * or /
        r'[+\-*/]$',     # Expression ending with an operator
        r'\(\)',         # Empty parentheses
        r'[+\-*/]\)',    # Operator before closing parenthesis
        r'\([+*/]',      # Opening parenthesis followed by operator (except -)
        r'\)\(',         # Adjacent parentheses without operator
        r'\d\(',         # Number directly before opening parenthesis
        r'\)\d',         # Closing parenthesis directly before number
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, expression):
            return False
    
    # Additional check for ** (power operator) - allow it
    expression_check = expression.replace('**', 'P')  # Temporarily replace ** to avoid false positive
    if re.search(r'[+\-*/]{2,}', expression_check):
        return False
    
    # Check for // (floor division) - allow it
    expression_check = expression.replace('//', 'F')  # Temporarily replace // to avoid false positive
    if re.search(r'[+\-*/]{2,}', expression_check):
        return False
    
    return True


def evaluate_safe_expression(expression):
    """
    Safely evaluates a mathematical expression after validation.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the evaluation, or an error message
    """
    # First validate the expression
    if not validate_math_expression(expression):
        return "Error: Invalid expression format - potential security risk"
    
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # Python 3.8+
            ast.Num,       # Older Python versions
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
        }
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Handle compatibility for different Python versions
                if hasattr(ast, 'Num') and isinstance(node, ast.Num):
                    continue
                if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
                    continue
                return f"Error: Unsafe operation detected - potential security risk"
        
        # If validation passes, compile and evaluate
        compiled = compile(tree, '<string>', 'eval')
        
        # Create a restricted namespace with only math operations
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
        }
        
        # Evaluate the expression
        result = eval(compiled, safe_dict)
        
        # Check for division by zero
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            return "Error: Division by zero"
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid syntax - potential security risk"
    except Exception as e:
        return f"Error: Evaluation failed - potential security risk"


# Example usage and test cases
if __name__ == "__main__":
    test_expressions = [
        ("2 + 3", True),
        ("10 * (5 - 3)", True),
        ("3.14 * 2", True),
        ("2 ** 3", True),
        ("10 // 3", True),
        ("5 + + 3", False),
        ("(2 + 3", False),
        ("2 + 3)", False),
        ("2 +* 3", False),
        ("", False),
        ("2 + abc", False),
        ("2 + 3; print('hack')", False),
        ("__import__('os')", False),
        ("2(3)", False),
        ("(2)3", False),
        ("()", False),
        ("2 + ", False),
        ("* 5", False),
    ]
    
    print("Validation Tests:")
    for expr, expected in test_expressions:
        result = validate_math_expression(expr)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{expr}' -> {result} (expected {expected})")
    
    print("\nEvaluation Tests:")
    eval_tests = [
        "2 + 3",
        "10 * (5 - 3)",
        "3.14 * 2",
        "2 ** 3",
        "10 // 3",
        "10 / 3",
        "10 % 3",
        "-5 + 3",
        "(2 + 3) * (4 - 1)",
        "2 + 3 * 4",
        "10 / 0",
        "__import__('os')",
        "print('hello')",
        "5 + + 3",
    ]
    
    for expr in eval_tests:
        result = evaluate_safe_expression(expr)
        print(f"'{expr}' -> {result}")
