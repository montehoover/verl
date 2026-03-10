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
    
    # Parse the SQL statement into components
    query_components = parse_sql_statement(sql_statement)
    
    # Create pipeline of operations
    pipeline = create_query_pipeline(query_components)
    
    # Execute pipeline on dataset
    return execute_pipeline(pipeline, dataset_records)


def parse_sql_statement(sql_statement):
    """Parse SQL statement into structured components."""
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
    
    return {
        'select': parse_select_fields(select_clause),
        'where': parse_where_conditions(where_clause) if where_clause else None,
        'order_by': {
            'field': order_clause,
            'descending': desc_order
        } if order_clause else None
    }


def parse_select_fields(select_clause):
    """Parse SELECT clause into field list."""
    if select_clause == '*':
        return None  # Select all fields
    return [field.strip() for field in select_clause.split(',')]


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


def create_query_pipeline(query_components):
    """Create a pipeline of operations from query components."""
    pipeline = []
    
    # Add WHERE filter if present
    if query_components['where']:
        pipeline.append(create_where_filter(query_components['where']))
    
    # Add ORDER BY sort if present
    if query_components['order_by']:
        pipeline.append(create_order_by_sort(
            query_components['order_by']['field'],
            query_components['order_by']['descending']
        ))
    
    # Add SELECT projection if needed
    if query_components['select'] is not None:
        pipeline.append(create_select_projection(query_components['select']))
    
    return pipeline


def create_where_filter(conditions):
    """Create a filter function for WHERE clause."""
    def filter_function(records):
        return [record for record in records if evaluate_conditions(record, conditions)]
    return filter_function


def create_order_by_sort(field, descending):
    """Create a sort function for ORDER BY clause."""
    def sort_function(records):
        try:
            return sorted(records, key=lambda x: x.get(field, ''), reverse=descending)
        except Exception as e:
            raise ValueError(f"Error sorting by field '{field}': {str(e)}")
    return sort_function


def create_select_projection(fields):
    """Create a projection function for SELECT clause."""
    def projection_function(records):
        result = []
        for record in records:
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
                else:
                    new_record[field] = None
            result.append(new_record)
        return result
    return projection_function


def execute_pipeline(pipeline, data):
    """Execute a pipeline of operations on data."""
    result = data.copy()
    for operation in pipeline:
        result = operation(result)
    return result


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
