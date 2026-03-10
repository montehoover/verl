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
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("Data must be a list of dictionaries")
    
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    # Parse the query
    query = query.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_clause = select_match.group(1).strip()
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.*?)$', query, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Process SELECT clause
    if select_clause == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
        # Validate field names
        if data:
            all_fields = set()
            for row in data:
                all_fields.update(row.keys())
            for field in selected_fields:
                if field not in all_fields:
                    raise ValueError(f"Invalid field in SELECT: {field}")
    
    # Apply WHERE clause
    filtered_data = data
    if where_clause:
        filtered_data = []
        for row in data:
            if evaluate_where_clause(row, where_clause):
                filtered_data.append(row)
    
    # Apply SELECT clause
    if selected_fields is None:
        result_data = [dict(row) for row in filtered_data]
    else:
        result_data = []
        for row in filtered_data:
            new_row = {}
            for field in selected_fields:
                if field in row:
                    new_row[field] = row[field]
            result_data.append(new_row)
    
    # Apply ORDER BY clause
    if order_clause:
        result_data = apply_order_by(result_data, order_clause)
    
    return result_data


def evaluate_where_clause(row, where_clause):
    """Evaluate WHERE clause for a single row"""
    # Parse conditions (supports AND only for simplicity)
    conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
    
    for condition in conditions:
        # Parse single condition
        match = re.match(r'(\w+)\s*([><=!]+)\s*(.+)', condition.strip())
        if not match:
            raise ValueError(f"Invalid WHERE condition: {condition}")
        
        field, op, value_str = match.groups()
        
        if field not in row:
            return False
        
        # Parse value
        value = parse_value(value_str.strip())
        row_value = row[field]
        
        # Apply operator
        if op == '=':
            if not (row_value == value):
                return False
        elif op == '!=':
            if not (row_value != value):
                return False
        elif op == '>':
            if not (row_value > value):
                return False
        elif op == '<':
            if not (row_value < value):
                return False
        elif op == '>=':
            if not (row_value >= value):
                return False
        elif op == '<=':
            if not (row_value <= value):
                return False
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    return True


def parse_value(value_str):
    """Parse a value from string format"""
    # Remove quotes if present
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    
    # Try to parse as number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return value_str


def apply_order_by(data, order_clause):
    """Apply ORDER BY clause to data"""
    # Parse ORDER BY clause
    parts = order_clause.split()
    if not parts:
        raise ValueError("Invalid ORDER BY clause")
    
    field = parts[0]
    ascending = True
    
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            ascending = False
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Check if field exists
    if data and field not in data[0]:
        raise ValueError(f"Invalid field in ORDER BY: {field}")
    
    # Sort data
    return sorted(data, key=lambda x: x.get(field, ''), reverse=not ascending)
