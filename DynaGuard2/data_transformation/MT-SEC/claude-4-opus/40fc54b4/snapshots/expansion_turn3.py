import re
import ast
import operator

def is_valid_expression(expression):
    """
    Check if a string is a valid arithmetic expression using only +, -, *, / operations.
    
    Args:
        expression: String representing the arithmetic expression
        
    Returns:
        Boolean indicating if the expression is valid
    """
    # Remove all whitespace
    expression = expression.replace(" ", "")
    
    # Empty string is not valid
    if not expression:
        return False
    
    # Check for invalid characters - only allow digits, operators, parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        return False
    
    # Check for empty parentheses
    if "()" in expression:
        return False
    
    # Check for consecutive operators
    if re.search(r'[+\-*/]{2,}', expression):
        return False
    
    # Check if expression starts or ends with an operator (except - at start)
    if re.match(r'^[+*/]', expression) or re.search(r'[+\-*/]$', expression):
        return False
    
    # Check for operators before/after parentheses incorrectly
    if re.search(r'\)[0-9.]', expression) or re.search(r'[0-9.]\(', expression):
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
    
    # Check for valid number format (no multiple decimal points in a number)
    numbers = re.findall(r'[0-9.]+', expression)
    for num in numbers:
        if num.count('.') > 1:
            return False
        if num == '.':
            return False
    
    # Try to evaluate to catch any remaining syntax errors
    try:
        # Replace the expression with a safe version for validation
        # This is just to check syntax, not to actually evaluate
        test_expr = expression
        # Ensure no division by zero for validation
        test_expr = re.sub(r'/\s*0+(?![0-9.])', '/1', test_expr)
        compile(test_expr, '<string>', 'eval')
        return True
    except:
        return False


def calculate_expression(expression):
    """
    Safely calculate the result of a validated arithmetic expression.
    
    Args:
        expression: String representing the arithmetic expression
        
    Returns:
        The calculated result (float) or an error message (string)
    """
    # First validate the expression
    if not is_valid_expression(expression):
        return "Error: Invalid expression format"
    
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }
    
    class SafeEvaluator(ast.NodeVisitor):
        def visit(self, node):
            if type(node) not in [ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant]:
                if type(node).__name__ not in ['Add', 'Sub', 'Mult', 'Div', 'USub', 'UAdd']:
                    raise ValueError(f"Unsafe operation: {type(node).__name__}")
            return super().visit(node)
        
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operation not allowed: {op_type.__name__}")
            # Check for division by zero
            if op_type == ast.Div and right == 0:
                raise ValueError("Division by zero")
            return allowed_operators[op_type](left, right)
        
        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operation not allowed: {op_type.__name__}")
            return allowed_operators[op_type](operand)
        
        def visit_Num(self, node):  # Python < 3.8
            return node.n
        
        def visit_Constant(self, node):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate safely
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree.body)
        
        return float(result)
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception:
        return "Error: Invalid expression"


def evaluate_arith_expression(input_expr):
    """
    Evaluate an arithmetic expression and return the result as a string.
    
    Args:
        input_expr: String representing the arithmetic expression to evaluate
        
    Returns:
        String containing the result or 'Computation Error!' if evaluation fails
    """
    try:
        # Use the calculate_expression function to safely evaluate
        result = calculate_expression(input_expr)
        
        # Check if the result is an error message
        if isinstance(result, str) and result.startswith("Error:"):
            return "Computation Error!"
        
        # Convert the numeric result to string
        return str(result)
        
    except Exception:
        return "Computation Error!"
