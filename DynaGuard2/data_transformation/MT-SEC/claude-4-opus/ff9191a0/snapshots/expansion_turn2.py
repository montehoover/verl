import re
from typing import List, Dict, Union

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

def substitute_variables(tokens: List[str], variables: Dict[str, Union[int, float]]) -> List[str]:
    """
    Substitute variables in a list of tokens with their corresponding values.
    
    Args:
        tokens (List[str]): List of tokens from parsed expression
        variables (Dict[str, Union[int, float]]): Dictionary mapping variable names to values
        
    Returns:
        List[str]: New list of tokens with variables replaced by their values
    """
    substituted_tokens = []
    
    for token in tokens:
        # Check if token is a variable name (starts with letter or underscore)
        if re.match(r'^[a-zA-Z_]\w*$', token):
            # If variable exists in the dictionary, substitute it
            if token in variables:
                substituted_tokens.append(str(variables[token]))
            else:
                # Keep the variable as is if not found in dictionary
                substituted_tokens.append(token)
        else:
            # Keep operators and numbers as is
            substituted_tokens.append(token)
    
    return substituted_tokens
