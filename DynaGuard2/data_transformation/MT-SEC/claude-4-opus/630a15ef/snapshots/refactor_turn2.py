import re


def process_user_query(query: str):
    """
    Process a user-provided mathematical expression and compute its result.
    
    This function safely evaluates mathematical expressions by first validating
    the input to ensure it contains only allowed characters and is properly
    formed. It's designed for use in web-based calculators where user input
    needs to be carefully validated before evaluation.
    
    Args:
        query (str): The user-provided mathematical expression to evaluate.
                    Should contain only numbers, basic operators (+, -, *, /),
                    parentheses, decimal points, and whitespace.
    
    Returns:
        float or int: The computed result of the mathematical expression.
    
    Raises:
        ValueError: If the input contains unsafe characters, has unbalanced
                   parentheses, is empty, or if the expression is malformed.
    
    Examples:
        >>> process_user_query("2 + 3 * 4")
        14
        >>> process_user_query("(10 - 5) / 2")
        2.5
    """
    # Validate input contains only safe characters
    # Allowed: digits, basic math operators, parentheses, decimal points, whitespace
    if re.search(r'[^0-9+\-*/().\s]', query):
        raise ValueError("Expression contains unsafe characters")
    
    # Validate parentheses are properly balanced
    paren_count = 0
    for char in query:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            
        # Early check for closing parenthesis without matching opening
        if paren_count < 0:
            raise ValueError("Expression has unbalanced parentheses")
    
    # Final check for unclosed opening parentheses
    if paren_count != 0:
        raise ValueError("Expression has unbalanced parentheses")
    
    # Validate expression is not empty
    if not query.strip():
        raise ValueError("Expression is empty")
    
    # Attempt to evaluate the validated expression
    try:
        result = eval(query)
        return result
    except (SyntaxError, NameError, ZeroDivisionError) as e:
        raise ValueError(f"Expression is malformed: {str(e)}")
