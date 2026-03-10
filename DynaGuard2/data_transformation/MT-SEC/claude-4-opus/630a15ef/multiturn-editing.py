import re

def simple_calculate(num1, num2, operator):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1 (float): First number
        num2 (float): Second number
        operator (str): Operation to perform ('+', '-', '*', '/')
    
    Returns:
        float: Result of the operation
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}")


def simple_calculate(expression):
    """
    Parse and calculate a simple arithmetic expression string.
    
    Args:
        expression (str): String containing a simple arithmetic operation (e.g., '7 + 8')
    
    Returns:
        float: Result of the operation
    """
    # Remove extra whitespace
    expression = expression.strip()
    
    # Try to find the operator
    operators = ['+', '-', '*', '/']
    operator = None
    operator_index = -1
    
    for op in operators:
        # Find the last occurrence to handle negative numbers
        index = expression.rfind(op)
        if index > 0:  # Ensure it's not at the beginning (negative number)
            operator = op
            operator_index = index
            break
    
    if operator is None:
        raise ValueError("No valid operator found in expression")
    
    # Split the expression
    num1_str = expression[:operator_index].strip()
    num2_str = expression[operator_index + 1:].strip()
    
    # Convert to float
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError("Invalid number format in expression")
    
    # Perform calculation
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2


def process_user_query(query):
    """
    Process a mathematical expression provided by a user.
    
    Args:
        query (str): Mathematical expression to evaluate
    
    Returns:
        float: Computed result of the expression
    
    Raises:
        ValueError: If the input contains unsafe characters or is malformed
    """
    # Strip whitespace
    query = query.strip()
    
    # Check for empty query
    if not query:
        raise ValueError("Empty query provided")
    
    # Define allowed characters: digits, operators, parentheses, decimal point, and whitespace
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if query contains only allowed characters
    if not re.match(allowed_pattern, query):
        raise ValueError("Query contains unsafe characters")
    
    # Additional safety checks
    # Check for common injection patterns
    dangerous_patterns = [
        r'__',  # Double underscore (Python internals)
        r'import',  # Import statements
        r'exec',  # Exec function
        r'eval',  # Eval function (even though we might use it)
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # Input function
        r'raw_input',  # Raw input function
        r'compile',  # Compile function
        r'globals',  # Globals function
        r'locals',  # Locals function
        r'vars',  # Vars function
        r'dir',  # Dir function
    ]
    
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            raise ValueError(f"Query contains unsafe pattern: {pattern}")
    
    # Check for balanced parentheses
    paren_count = 0
    for char in query:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            raise ValueError("Unbalanced parentheses in query")
    
    if paren_count != 0:
        raise ValueError("Unbalanced parentheses in query")
    
    # Check for valid operator placement
    # Operators should not be at the beginning or end (except minus at the beginning)
    operators = ['+', '*', '/']
    if query.endswith(tuple(operators)) or query.endswith('-'):
        raise ValueError("Query ends with an operator")
    
    if query.startswith(tuple(operators)):
        raise ValueError("Query starts with an invalid operator")
    
    # Check for consecutive operators (except for negative numbers)
    operator_pattern = r'[+\-*/]{2,}'
    if re.search(operator_pattern, query):
        # Allow for patterns like "5 * -3" but not "5 ** 3"
        matches = re.findall(operator_pattern, query)
        for match in matches:
            if match != '+-' and match != '--' and match != '*-' and match != '/-':
                raise ValueError("Query contains consecutive operators")
    
    try:
        # Use eval to compute the result
        # Since we've validated the input, this should be safe
        result = eval(query)
        
        # Ensure the result is a number
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
        
        # Convert to float for consistency
        return float(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression")
    except Exception as e:
        raise ValueError(f"Malformed expression: {str(e)}")
