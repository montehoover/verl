import re

def process_user_query(query: str):
    """
    Process a user query to compute a given mathematical expression.
    
    Args:
        query: str - the user-provided mathematical expression
        
    Returns:
        The computed result of the expression
        
    Raises:
        ValueError: if the input contains unsafe characters or if the expression is malformed
    """
    # Define a pattern for allowed characters: digits, operators, parentheses, spaces, and decimal points
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if the query contains only allowed characters
    if not re.match(allowed_pattern, query):
        raise ValueError("Expression contains unsafe characters")
    
    # Strip whitespace
    query = query.strip()
    
    # Check for empty expression
    if not query:
        raise ValueError("Expression is empty")
    
    try:
        # Create a restricted environment for eval
        # This prevents access to built-in functions and modules
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round
        }
        
        # Evaluate the expression safely
        result = eval(query, safe_dict)
        
        # Return the result
        return result
        
    except (SyntaxError, NameError, TypeError) as e:
        raise ValueError(f"Expression is malformed: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
