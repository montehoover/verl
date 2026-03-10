import re
import operator
from functools import partial

def run_sql_query(dataset, sql_query):
    """
    Process a custom SQL-like query on data represented as a list of dictionaries.
    
    Args:
        dataset: A list of dictionaries where each dictionary represents a record
        sql_query: A string containing the SQL-like query to be executed
        
    Returns:
        A list containing dictionaries that represent the results of the executed query
        
    Raises:
        ValueError: Raised if the query is not correctly formed or execution fails
    """
    if not dataset:
        return []
    
    # Parse the query
    query_upper = sql_query.upper()
    
    # Extract SELECT clause
    select_match = re.search(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_clause = sql_query[select_match.start(1):select_match.end(1)].strip()
    
    # Extract WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    where_clause = None
    if where_match:
        where_clause = sql_query[where_match.start(1):where_match.end(1)].strip()
    
    # Extract ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(.+?)$', query_upper, re.IGNORECASE)
    order_clause = None
    if order_match:
        order_clause = sql_query[order_match.start(1):order_match.end(1)].strip()
    
    # Process SELECT clause
    if select_clause.strip() == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    # Apply WHERE clause
    result = dataset.copy()
    if where_clause:
        result = apply_where_clause(result, where_clause)
    
    # Apply SELECT clause
    if selected_fields:
        result = apply_select_clause(result, selected_fields)
    
    # Apply ORDER BY clause
    if order_clause:
        result = apply_order_by_clause(result, order_clause)
    
    return result


def apply_where_clause(data, where_clause):
    """Apply WHERE clause filtering to the dataset."""
    # Parse conditions (supports simple AND/OR logic)
    conditions = parse_where_conditions(where_clause)
    
    filtered_data = []
    for record in data:
        if evaluate_conditions(record, conditions):
            filtered_data.append(record)
    
    return filtered_data


def parse_where_conditions(where_clause):
    """Parse WHERE clause into a list of conditions."""
    # Split by AND/OR (simple parsing)
    conditions = []
    
    # Replace AND/OR with markers for splitting
    clause = where_clause.replace(' AND ', ' __AND__ ').replace(' OR ', ' __OR__ ')
    clause = clause.replace(' and ', ' __AND__ ').replace(' or ', ' __OR__ ')
    
    parts = re.split(r'\s+__(?:AND|OR)__\s+', clause)
    
    for i, part in enumerate(parts):
        # Parse individual condition
        match = re.match(r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)', part.strip())
        if not match:
            raise ValueError(f"Invalid WHERE condition: {part}")
        
        field, op, value = match.groups()
        value = value.strip().strip("'\"")
        
        # Try to convert to number if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        # Determine logical operator for this condition
        if i > 0:
            # Check what came before this condition
            before_idx = where_clause.upper().find(parts[i-1].upper()) + len(parts[i-1])
            after_idx = where_clause.upper().find(part.upper())
            between = where_clause[before_idx:after_idx].upper()
            
            if ' AND ' in between:
                logical_op = 'AND'
            else:
                logical_op = 'OR'
        else:
            logical_op = None
        
        conditions.append({
            'field': field,
            'operator': op,
            'value': value,
            'logical_op': logical_op
        })
    
    return conditions


def evaluate_conditions(record, conditions):
    """Evaluate conditions against a record."""
    if not conditions:
        return True
    
    # Map string operators to functions
    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    result = None
    
    for i, cond in enumerate(conditions):
        field_value = record.get(cond['field'])
        if field_value is None:
            cond_result = False
        else:
            try:
                cond_result = ops[cond['operator']](field_value, cond['value'])
            except:
                cond_result = False
        
        if i == 0:
            result = cond_result
        elif cond['logical_op'] == 'AND':
            result = result and cond_result
        elif cond['logical_op'] == 'OR':
            result = result or cond_result
    
    return result


def apply_select_clause(data, selected_fields):
    """Apply SELECT clause to project only specified fields."""
    result = []
    
    for record in data:
        new_record = {}
        for field in selected_fields:
            if field in record:
                new_record[field] = record[field]
            else:
                raise ValueError(f"Field '{field}' not found in dataset")
        result.append(new_record)
    
    return result


def apply_order_by_clause(data, order_clause):
    """Apply ORDER BY clause to sort the data."""
    # Parse ORDER BY clause
    parts = order_clause.split(',')
    sort_keys = []
    
    for part in parts:
        part = part.strip()
        if ' DESC' in part.upper():
            field = part[:part.upper().rfind(' DESC')].strip()
            reverse = True
        elif ' ASC' in part.upper():
            field = part[:part.upper().rfind(' ASC')].strip()
            reverse = False
        else:
            field = part
            reverse = False
        
        sort_keys.append((field, reverse))
    
    # Sort by multiple keys
    result = data.copy()
    for field, reverse in reversed(sort_keys):
        try:
            result.sort(key=lambda x: x.get(field, ''), reverse=reverse)
        except Exception as e:
            raise ValueError(f"Error sorting by field '{field}': {str(e)}")
    
    return result
