import re
import ast

def sanitize_expression(expression):
    """
    Check if a given string only contains numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows: digits, decimal points, operators (+, -, *, /), parentheses, and spaces
    pattern = r'^[0-9\+\-\*/\(\)\.\s]+$'
    
    if not expression or not isinstance(expression, str):
        return False
    
    # Check if the expression matches the allowed pattern
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to prevent empty or whitespace-only strings
    if not expression.strip():
        return False
    
    return True


def parse_expression(expression):
    """
    Parse a valid expression string into a structured representation.
    Returns a nested structure that respects operator precedence.
    
    Args:
        expression (str): The expression string to parse
        
    Returns:
        The parsed expression structure (can be a number, or a dict with 'op', 'left', 'right')
    """
    # Remove all spaces
    expression = expression.replace(' ', '')
    
    # Helper function to find matching closing parenthesis
    def find_matching_paren(expr, start):
        count = 1
        i = start + 1
        while i < len(expr) and count > 0:
            if expr[i] == '(':
                count += 1
            elif expr[i] == ')':
                count -= 1
            i += 1
        return i - 1
    
    # Helper function to parse expression recursively
    def parse_expr(expr):
        if not expr:
            return None
            
        # Handle parentheses first
        if expr[0] == '(':
            end = find_matching_paren(expr, 0)
            if end == len(expr) - 1:
                # Entire expression is wrapped in parentheses
                return parse_expr(expr[1:-1])
        
        # Find operators with lowest precedence (+ and -) from right to left
        paren_depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')':
                paren_depth += 1
            elif expr[i] == '(':
                paren_depth -= 1
            elif paren_depth == 0 and expr[i] in '+-':
                # Found a + or - operator at top level
                left = parse_expr(expr[:i])
                right = parse_expr(expr[i+1:])
                return {'op': expr[i], 'left': left, 'right': right}
        
        # Find operators with higher precedence (* and /) from right to left
        paren_depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')':
                paren_depth += 1
            elif expr[i] == '(':
                paren_depth -= 1
            elif paren_depth == 0 and expr[i] in '*/':
                # Found a * or / operator at top level
                left = parse_expr(expr[:i])
                right = parse_expr(expr[i+1:])
                return {'op': expr[i], 'left': left, 'right': right}
        
        # If no operators found, try to parse as a number
        try:
            return float(expr) if '.' in expr else int(expr)
        except ValueError:
            # Handle parentheses in the middle of expression
            if '(' in expr:
                start = expr.index('(')
                end = find_matching_paren(expr, start)
                # This shouldn't happen in a well-formed expression
                return None
            return None
    
    return parse_expr(expression)


def evaluate_and_compute(expression):
    """
    Validate, parse, and evaluate a mathematical expression string.
    
    Args:
        expression (str): The expression string to evaluate
        
    Returns:
        float or int: The computed result
        
    Raises:
        ValueError: If the input is invalid
    """
    # Validate the expression
    if not sanitize_expression(expression):
        raise ValueError("Invalid expression: contains disallowed characters or is empty")
    
    # Parse the expression
    parsed = parse_expression(expression)
    
    if parsed is None:
        raise ValueError("Invalid expression: failed to parse")
    
    # Helper function to evaluate the parsed structure
    def evaluate_parsed(node):
        if node is None:
            raise ValueError("Invalid expression: encountered None during evaluation")
        
        # If it's a number, return it
        if isinstance(node, (int, float)):
            return node
        
        # If it's an operation, evaluate left and right, then apply the operator
        if isinstance(node, dict) and 'op' in node:
            left_val = evaluate_parsed(node['left'])
            right_val = evaluate_parsed(node['right'])
            
            if node['op'] == '+':
                return left_val + right_val
            elif node['op'] == '-':
                return left_val - right_val
            elif node['op'] == '*':
                return left_val * right_val
            elif node['op'] == '/':
                if right_val == 0:
                    raise ValueError("Division by zero")
                return left_val / right_val
        
        raise ValueError("Invalid expression: unknown node type")
    
    # Evaluate and return the result
    result = evaluate_parsed(parsed)
    
    # Return int if the result is a whole number
    if isinstance(result, float) and result.is_integer():
        return int(result)
    
    return result
