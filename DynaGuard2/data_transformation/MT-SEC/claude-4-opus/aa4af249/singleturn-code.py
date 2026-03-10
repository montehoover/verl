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
    
    # Parse the SQL statement
    sql_statement = sql_statement.strip()
    
    # Extract SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_statement, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid SQL statement: Missing SELECT clause")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_statement, re.IGNORECASE)
    where_conditions = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(.*?)$', sql_statement, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Start with all records
    result = dataset_records.copy()
    
    # Apply WHERE clause
    if where_conditions:
        result = apply_where_clause(result, where_conditions)
    
    # Apply ORDER BY clause
    if order_clause:
        result = apply_order_by_clause(result, order_clause)
    
    # Apply SELECT clause
    if select_fields[0] != '*':
        result = apply_select_clause(result, select_fields)
    
    return result


def apply_where_clause(records, conditions):
    """Apply WHERE conditions to filter records."""
    # Parse comparison operators
    comparison_ops = {
        '>=': operator.ge,
        '<=': operator.le,
        '!=': operator.ne,
        '=': operator.eq,
        '>': operator.gt,
        '<': operator.lt
    }
    
    # Find the operator and split the condition
    for op_str, op_func in comparison_ops.items():
        if op_str in conditions:
            parts = conditions.split(op_str)
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip()
                
                # Try to convert value to appropriate type
                try:
                    # Try integer first
                    value = int(value)
                except ValueError:
                    try:
                        # Try float
                        value = float(value)
                    except ValueError:
                        # Keep as string, remove quotes if present
                        value = value.strip('\'"')
                
                # Filter records
                filtered = []
                for record in records:
                    if field in record:
                        try:
                            if op_func(record[field], value):
                                filtered.append(record)
                        except Exception:
                            # Skip records where comparison fails
                            pass
                
                return filtered
    
    raise ValueError(f"Invalid WHERE clause: {conditions}")


def apply_order_by_clause(records, order_clause):
    """Apply ORDER BY clause to sort records."""
    # Parse ORDER BY clause
    parts = order_clause.split()
    if not parts:
        raise ValueError("Invalid ORDER BY clause")
    
    field = parts[0]
    reverse = False
    
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            reverse = True
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Sort records
    try:
        return sorted(records, key=lambda x: x.get(field, ''), reverse=reverse)
    except Exception as e:
        raise ValueError(f"Error in ORDER BY clause: {str(e)}")


def apply_select_clause(records, fields):
    """Apply SELECT clause to filter fields in records."""
    result = []
    for record in records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        result.append(new_record)
    return result
