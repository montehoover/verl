import re

def process_user_query(query: str):
    # Check for unsafe characters
    if re.search(r'[^0-9+\-*/().\s]', query):
        raise ValueError("Expression contains unsafe characters")
    
    # Check for balanced parentheses
    paren_count = 0
    for char in query:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            raise ValueError("Expression has unbalanced parentheses")
    
    if paren_count != 0:
        raise ValueError("Expression has unbalanced parentheses")
    
    # Check for empty expression
    if not query.strip():
        raise ValueError("Expression is empty")
    
    try:
        # Evaluate the expression
        result = eval(query)
        return result
    except (SyntaxError, NameError, ZeroDivisionError) as e:
        raise ValueError(f"Expression is malformed: {str(e)}")
