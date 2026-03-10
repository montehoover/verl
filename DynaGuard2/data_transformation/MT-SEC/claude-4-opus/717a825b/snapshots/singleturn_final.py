import re
import operator
from functools import partial

def execute_custom_query(data, query):
    """
    Execute a custom SQL-like query on a list of dictionaries.
    
    Args:
        data: list of dictionaries representing the dataset
        query: str, a SQL-like query string
        
    Returns:
        list of dictionaries representing the query results
        
    Raises:
        ValueError: Raised when the query is invalid or cannot be executed
    """
    
    # Parse the query
    query = query.strip()
    
    # Regular expressions for parsing different parts of the query
    select_pattern = r'SELECT\s+(.*?)\s+FROM'
    from_pattern = r'FROM\s+(\w+)'
    where_pattern = r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)'
    order_by_pattern = r'ORDER\s+BY\s+(.*?)(?:\s+DESC|$)'
    desc_pattern = r'ORDER\s+BY\s+.*?\s+DESC'
    
    # Extract SELECT fields
    select_match = re.search(select_pattern, query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract FROM table (we'll ignore it since we're working with the provided data)
    from_match = re.search(from_pattern, query, re.IGNORECASE)
    if not from_match:
        raise ValueError("Invalid query: FROM clause not found")
    
    # Start with all data
    result = data[:]
    
    # Apply WHERE clause if present
    where_match = re.search(where_pattern, query, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1).strip()
        result = apply_where_clause(result, where_clause)
    
    # Apply ORDER BY clause if present
    order_by_match = re.search(order_by_pattern, query, re.IGNORECASE)
    if order_by_match:
        order_field = order_by_match.group(1).strip()
        desc = bool(re.search(desc_pattern, query, re.IGNORECASE))
        result = apply_order_by(result, order_field, desc)
    
    # Apply SELECT clause
    if select_fields != ['*']:
        result = apply_select(result, select_fields)
    
    return result


def apply_where_clause(data, where_clause):
    """Apply WHERE clause filtering to the data."""
    # Parse the WHERE clause
    # Support simple conditions: field operator value
    # Operators: =, !=, <, >, <=, >=
    
    # Pattern for parsing conditions
    condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)'
    match = re.match(condition_pattern, where_clause.strip())
    
    if not match:
        raise ValueError(f"Invalid WHERE clause: {where_clause}")
    
    field, op, value = match.groups()
    
    # Determine the operator function
    op_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported operator: {op}")
    
    op_func = op_map[op]
    
    # Parse the value
    value = value.strip()
    
    # Remove quotes if present
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    else:
        # Try to convert to number
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  # Keep as string
    
    # Filter the data
    filtered_data = []
    for row in data:
        if field not in row:
            raise ValueError(f"Field '{field}' not found in data")
        
        try:
            if op_func(row[field], value):
                filtered_data.append(row)
        except TypeError:
            # Type mismatch, skip this row
            pass
    
    return filtered_data


def apply_order_by(data, field, descending=False):
    """Apply ORDER BY clause to sort the data."""
    if not data:
        return data
    
    # Check if field exists
    if field not in data[0]:
        raise ValueError(f"Field '{field}' not found in data")
    
    # Sort the data
    try:
        sorted_data = sorted(data, key=lambda x: x[field], reverse=descending)
    except TypeError:
        # Mixed types, convert to string for comparison
        sorted_data = sorted(data, key=lambda x: str(x[field]), reverse=descending)
    
    return sorted_data


def apply_select(data, fields):
    """Apply SELECT clause to project specific fields."""
    if not data:
        return data
    
    # Check if all fields exist
    for field in fields:
        if field not in data[0]:
            raise ValueError(f"Field '{field}' not found in data")
    
    # Project the fields
    result = []
    for row in data:
        new_row = {field: row[field] for field in fields}
        result.append(new_row)
    
    return result
