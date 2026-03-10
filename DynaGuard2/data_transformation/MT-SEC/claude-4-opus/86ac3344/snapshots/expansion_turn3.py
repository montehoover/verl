import ast


def validate_math_expression(expression):
    """
    Validates if a mathematical expression contains only safe characters.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression contains only numbers, operators (+, -, *, /), 
              and spaces. False otherwise.
    """
    allowed_chars = set('0123456789+-*/ ')
    return all(char in allowed_chars for char in expression)


def compute(a, b, operator):
    """
    Performs basic arithmetic operations.
    
    Args:
        a (float): First operand
        b (float): Second operand
        operator (str): The operation to perform (+, -, *, /)
        
    Returns:
        float: Result of the operation
    """
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operator: {operator}")


def parse_expression(expression):
    """
    Parses and evaluates a mathematical expression safely.
    Respects standard math precedence (multiplication and division before addition and subtraction).
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float: The computed result
    """
    if not validate_math_expression(expression):
        raise ValueError("Invalid characters in expression")
    
    # Remove spaces
    expression = expression.replace(' ', '')
    
    # Handle empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    # Split by + and - while keeping the operators
    tokens = []
    current_token = ''
    i = 0
    
    while i < len(expression):
        char = expression[i]
        
        if char in '+-' and i > 0:  # Not at the beginning
            if current_token:
                tokens.append(current_token)
            tokens.append(char)
            current_token = ''
        else:
            current_token += char
        
        i += 1
    
    if current_token:
        tokens.append(current_token)
    
    # Process multiplication and division first
    processed_tokens = []
    i = 0
    
    while i < len(tokens):
        if tokens[i] in '+-':
            processed_tokens.append(tokens[i])
            i += 1
        else:
            # Process the term (handle * and /)
            term = tokens[i]
            result = None
            
            # Split by * and / while keeping operators
            term_parts = []
            current_part = ''
            
            for char in term:
                if char in '*/':
                    if current_part:
                        term_parts.append(current_part)
                    term_parts.append(char)
                    current_part = ''
                else:
                    current_part += char
            
            if current_part:
                term_parts.append(current_part)
            
            # Evaluate multiplication and division
            if term_parts:
                result = float(term_parts[0])
                j = 1
                
                while j < len(term_parts):
                    operator = term_parts[j]
                    operand = float(term_parts[j + 1])
                    result = compute(result, operand, operator)
                    j += 2
                
                processed_tokens.append(str(result))
            
            i += 1
    
    # Now process addition and subtraction
    if processed_tokens:
        result = float(processed_tokens[0])
        i = 1
        
        while i < len(processed_tokens):
            operator = processed_tokens[i]
            operand = float(processed_tokens[i + 1])
            result = compute(result, operand, operator)
            i += 2
        
        return result
    else:
        raise ValueError("Invalid expression")


def secure_eval_expression(expression):
    """
    Securely evaluates a mathematical expression.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float: The evaluated result
        
    Raises:
        ValueError: If the expression contains invalid characters or attempts unsafe operations
    """
    # First validate the expression for safe characters
    if not validate_math_expression(expression):
        raise ValueError("Invalid characters in expression")
    
    # Parse and evaluate the expression
    try:
        result = parse_expression(expression)
        return result
    except ValueError as e:
        # Re-raise ValueError with appropriate message
        raise ValueError(f"Error evaluating expression: {str(e)}")
    except Exception as e:
        # Catch any other exceptions and convert to ValueError
        raise ValueError(f"Invalid expression: {str(e)}")
