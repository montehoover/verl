import re

def is_valid_expression(expression):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows:
    # - digits (0-9)
    # - decimal points
    # - arithmetic operators: +, -, *, /
    # - parentheses: ( )
    # - spaces
    pattern = r'^[0-9\+\-\*/\(\)\.\s]+$'
    
    if not expression or not isinstance(expression, str):
        return False
    
    # Check if the expression matches the allowed pattern
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to ensure proper structure
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[\+\*/]{2,}', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[\+\*/]', expression):
        return False
    
    # Check for operators at the end
    if re.search(r'[\+\-\*/]$', expression):
        return False
    
    return True


def tokenize(expression):
    """
    Tokenizes a mathematical expression into numbers and operators.
    
    Args:
        expression (str): The expression to tokenize
        
    Returns:
        list: List of tokens (numbers and operators)
    """
    # Remove spaces
    expression = expression.replace(' ', '')
    
    tokens = []
    current_number = ''
    
    i = 0
    while i < len(expression):
        char = expression[i]
        
        if char.isdigit() or char == '.':
            current_number += char
        else:
            if current_number:
                tokens.append(float(current_number) if '.' in current_number else int(current_number))
                current_number = ''
            
            if char in '+-*/()':
                tokens.append(char)
        
        i += 1
    
    if current_number:
        tokens.append(float(current_number) if '.' in current_number else int(current_number))
    
    return tokens


def parse_expression(expression):
    """
    Parses a mathematical expression into a tree structure that respects operator precedence.
    Returns a nested list structure where each operator node is [operator, left, right].
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        list/number: Tree structure representing the parsed expression
    """
    if not is_valid_expression(expression):
        raise ValueError("Invalid expression")
    
    tokens = tokenize(expression)
    
    def parse_tokens(tokens, start=0):
        """Parse tokens with operator precedence."""
        return parse_addition_subtraction(tokens, start)
    
    def parse_addition_subtraction(tokens, start):
        """Parse addition and subtraction (lowest precedence)."""
        left, pos = parse_multiplication_division(tokens, start)
        
        while pos < len(tokens) and tokens[pos] in ['+', '-']:
            op = tokens[pos]
            right, pos = parse_multiplication_division(tokens, pos + 1)
            left = [op, left, right]
        
        return left, pos
    
    def parse_multiplication_division(tokens, start):
        """Parse multiplication and division (higher precedence)."""
        left, pos = parse_parentheses(tokens, start)
        
        while pos < len(tokens) and tokens[pos] in ['*', '/']:
            op = tokens[pos]
            right, pos = parse_parentheses(tokens, pos + 1)
            left = [op, left, right]
        
        return left, pos
    
    def parse_parentheses(tokens, start):
        """Parse parentheses and numbers (highest precedence)."""
        if start >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token = tokens[start]
        
        if token == '(':
            # Find matching closing parenthesis
            level = 1
            end = start + 1
            while end < len(tokens) and level > 0:
                if tokens[end] == '(':
                    level += 1
                elif tokens[end] == ')':
                    level -= 1
                end += 1
            
            if level != 0:
                raise ValueError("Mismatched parentheses")
            
            # Parse the expression inside parentheses
            inner_result, _ = parse_tokens(tokens, start + 1)
            return inner_result, end
        
        elif token == ')':
            # This is handled by the parent call
            return None, start
        
        elif isinstance(token, (int, float)):
            return token, start + 1
        
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    result, _ = parse_tokens(tokens)
    return result
