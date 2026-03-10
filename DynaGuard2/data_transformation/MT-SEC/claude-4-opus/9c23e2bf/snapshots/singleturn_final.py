import re
import operator
from functools import partial

def run_custom_query(dataset, query):
    """
    Execute a basic SQL-like statement on a dataset.
    
    Args:
        dataset: A list of dictionaries representing data records
        query: A string containing the SQL-like statement
        
    Returns:
        A list of dictionaries containing the query results
        
    Raises:
        ValueError: If the query format is invalid or can't be processed
    """
    # Parse the query
    query = query.strip()
    
    # Regular expressions for parsing
    select_pattern = r'SELECT\s+(.*?)\s+FROM'
    where_pattern = r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)'
    order_pattern = r'ORDER\s+BY\s+(.*)$'
    
    # Extract SELECT clause
    select_match = re.search(select_pattern, query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query format: Missing SELECT clause")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract WHERE clause (optional)
    where_match = re.search(where_pattern, query, re.IGNORECASE)
    where_condition = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause (optional)
    order_match = re.search(order_pattern, query, re.IGNORECASE)
    order_field = order_match.group(1).strip() if order_match else None
    
    # Process the dataset
    result = dataset.copy()
    
    # Apply WHERE clause
    if where_condition:
        result = apply_where_clause(result, where_condition)
    
    # Apply ORDER BY clause
    if order_field:
        result = apply_order_by(result, order_field)
    
    # Apply SELECT clause
    result = apply_select_clause(result, select_fields)
    
    return result


def apply_where_clause(data, condition):
    """Apply WHERE filtering to the dataset."""
    # Parse the condition (support basic comparisons)
    # Pattern for field operator value
    condition_pattern = r'(\w+)\s*([><=]+)\s*(.+)'
    match = re.match(condition_pattern, condition.strip())
    
    if not match:
        raise ValueError(f"Invalid WHERE condition: {condition}")
    
    field, op, value = match.groups()
    
    # Determine the operator
    op_map = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '=': operator.eq,
        '==': operator.eq
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported operator: {op}")
    
    op_func = op_map[op]
    
    # Convert value to appropriate type
    try:
        # Try to convert to number
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        # Keep as string, remove quotes if present
        value = value.strip().strip('"\'')
    
    # Filter the data
    filtered = []
    for record in data:
        if field not in record:
            raise ValueError(f"Field '{field}' not found in dataset")
        
        try:
            if op_func(record[field], value):
                filtered.append(record)
        except TypeError:
            # Type mismatch in comparison
            continue
    
    return filtered


def apply_order_by(data, order_field):
    """Apply ORDER BY sorting to the dataset."""
    # Check for DESC/ASC
    parts = order_field.split()
    field = parts[0]
    descending = False
    
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            descending = True
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Verify field exists
    if data and field not in data[0]:
        raise ValueError(f"Field '{field}' not found in dataset")
    
    # Sort the data
    try:
        return sorted(data, key=lambda x: x[field], reverse=descending)
    except Exception as e:
        raise ValueError(f"Error sorting by field '{field}': {str(e)}")


def apply_select_clause(data, fields):
    """Apply SELECT projection to the dataset."""
    if not fields:
        raise ValueError("No fields specified in SELECT clause")
    
    # Handle SELECT *
    if len(fields) == 1 and fields[0] == '*':
        return data
    
    # Project specific fields
    result = []
    for record in data:
        projected = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in dataset")
            projected[field] = record[field]
        result.append(projected)
    
    return result
