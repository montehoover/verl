import re

def parse_math_expression(expression):
    """
    Parse a mathematical expression string and return a list of recognized mathematical elements.
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        list: A list of recognized mathematical elements (numbers and operators)
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Regular expression pattern to match numbers (including decimals) and operators
    # Matches: integers, decimals, and common mathematical operators
    pattern = r'(\d+\.?\d*|[\+\-\*/\(\)\^])'
    
    # Find all matches
    elements = re.findall(pattern, expression)
    
    # Filter out empty strings
    elements = [elem for elem in elements if elem]
    
    return elements


# Example usage
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3 * 4",
        "10.5 - 3.2",
        "(5 + 3) * 2",
        "2^3 + 1",
        "100/5-10"
    ]
    
    for expr in test_expressions:
        result = parse_math_expression(expr)
        print(f"Expression: {expr}")
        print(f"Parsed elements: {result}")
        print()
