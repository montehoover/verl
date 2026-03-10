import re
import operator
from functools import partial

def process_sql_request(dataset_records, sql_statement):
    """
    Process a custom SQL-like query on data represented as a list of dictionaries.
    
    Args:
        dataset_records: A list of dictionaries where each dictionary represents a record
        sql_statement: A string containing the SQL-like query to be executed
        
    Returns:
        A list containing dictionaries that represent the results of the executed query
        
    Raises:
        ValueError: Raised if the query is not correctly formed or execution fails
    """
    if not sql_statement or not isinstance(sql_statement, str):
        raise ValueError("SQL statement must be a non-empty string")
    
    if not isinstance(dataset_records, list):
        raise ValueError("Dataset records must be a list")
    
    # Normalize the SQL statement
    sql_statement = sql_statement.strip()
    
    # Parse SELECT clause
    select_match = re.match(r'^SELECT\s+(.+?)(?:\s+FROM|$)', sql_statement, re.IGNORECASE)
    if not select_match:
        raise ValueError("Query must start with SELECT")
    
    select_clause = select_match.group(1).strip()
    
    # Parse WHERE clause (optional)
    where_match = re.search(r'\sWHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', sql_statement, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Parse ORDER BY clause (optional)
    order_match = re.search(r'\sORDER\s+BY\s+(.+?)(?:\s+ASC|\s+DESC|$)', sql_statement, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Check for ASC/DESC
    desc_order = bool(re.search(r'\sDESC(?:\s|$)', sql_statement, re.IGNORECASE))
    
    # Process SELECT fields
    if select_clause == '*':
        select_fields = None  # Select all fields
    else:
        select_fields = [field.strip() for field in select_clause.split(',')]
        
    # Start with all records
    result = dataset_records.copy()
    
    # Apply WHERE clause
    if where_clause:
        result = apply_where_clause(result, where_clause)
    
    # Apply ORDER BY clause
    if order_clause:
        result = apply_order_by_clause(result, order_clause, desc_order)
    
    # Apply SELECT clause
    if select_fields:
        result = apply_select_clause(result, select_fields)
    
    return result


def apply_where_clause(records, where_clause):
    """Apply WHERE filtering to records."""
    filtered_records = []
    
    # Parse conditions (simple implementation supporting AND/OR)
    conditions = parse_where_conditions(where_clause)
    
    for record in records:
        if evaluate_conditions(record, conditions):
            filtered_records.append(record)
    
    return filtered_records


def parse_where_conditions(where_clause):
    """Parse WHERE clause into conditions."""
    # Split by AND/OR while preserving the operator
    parts = re.split(r'\s+(AND|OR)\s+', where_clause, flags=re.IGNORECASE)
    
    conditions = []
    current_operator = 'AND'  # Default operator
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # This is a condition
            condition = parse_single_condition(part.strip())
            conditions.append((current_operator, condition))
        else:  # This is an operator
            current_operator = part.upper()
    
    return conditions


def parse_single_condition(condition_str):
    """Parse a single condition like 'field = value'."""
    # Match comparison operators
    match = re.match(r'(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)', condition_str)
    if not match:
        raise ValueError(f"Invalid condition: {condition_str}")
    
    field = match.group(1)
    op = match.group(2)
    value_str = match.group(3).strip()
    
    # Parse value
    value = parse_value(value_str)
    
    # Map operators
    op_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge
    }
    
    return (field, op_map[op], value)


def parse_value(value_str):
    """Parse a value from string format."""
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


def evaluate_conditions(record, conditions):
    """Evaluate conditions against a record."""
    if not conditions:
        return True
    
    result = True
    current_operator = 'AND'
    
    for i, (op, (field, comparator, value)) in enumerate(conditions):
        if field not in record:
            condition_result = False
        else:
            try:
                condition_result = comparator(record[field], value)
            except:
                condition_result = False
        
        if i == 0:
            result = condition_result
        else:
            if conditions[i-1][0] == 'AND':
                result = result and condition_result
            else:  # OR
                result = result or condition_result
    
    return result


def apply_order_by_clause(records, order_field, desc_order):
    """Apply ORDER BY sorting to records."""
    try:
        return sorted(records, key=lambda x: x.get(order_field, ''), reverse=desc_order)
    except Exception as e:
        raise ValueError(f"Error sorting by field '{order_field}': {str(e)}")


def apply_select_clause(records, select_fields):
    """Apply SELECT projection to records."""
    result = []
    
    for record in records:
        new_record = {}
        for field in select_fields:
            if field in record:
                new_record[field] = record[field]
            else:
                # Field doesn't exist, include as None
                new_record[field] = None
        result.append(new_record)
    
    return result
