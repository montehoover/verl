import re

def parse_expression(expression):
    """
    Parse a mathematical expression into tokens.
    
    Args:
        expression (str): Mathematical expression to parse
        
    Returns:
        list: List of tokens (numbers, operators, variables)
    """
    # Define token patterns
    token_pattern = r'''
        (?P<NUMBER>\d+\.?\d*)|           # Integer or decimal number
        (?P<VARIABLE>[a-zA-Z_]\w*)|      # Variable names
        (?P<OPERATOR>[+\-*/^()])|        # Mathematical operators and parentheses
        (?P<WHITESPACE>\s+)              # Whitespace (to be ignored)
    '''
    
    # Compile the pattern with verbose flag for readability
    token_regex = re.compile(token_pattern, re.VERBOSE)
    
    tokens = []
    
    for match in token_regex.finditer(expression):
        token_type = match.lastgroup
        token_value = match.group()
        
        # Skip whitespace tokens
        if token_type != 'WHITESPACE':
            tokens.append(token_value)
    
    return tokens
