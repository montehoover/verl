import re
import operator
from functools import partial

def parse_select_clause(query, query_upper):
    """Parse the SELECT clause from the query."""
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query_upper)
    if not select_match:
        raise ValueError("Query must start with SELECT")
    
    select_clause = query[select_match.start(1):select_match.end(1)]
    
    if select_clause.strip() == '*':
        return None  # Select all fields
    else:
        return [field.strip() for field in select_clause.split(',')]

def parse_where_clause(query, query_upper):
    """Parse the WHERE clause from the query."""
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query_upper)
    where_conditions = []
    
    if where_match:
        where_clause = query[where_match.start(1):where_match.end(1)]
        condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*([\'"]?)(.+?)\3'
        
        for match in re.finditer(condition_pattern, where_clause):
            field, op, quote, value = match.groups()
            
            if not quote:  # No quotes, try to convert to number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            
            where_conditions.append((field, op, value))
    
    return where_conditions

def parse_order_by_clause(query, query_upper):
    """Parse the ORDER BY clause from the query."""
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query_upper)
    
    if order_match:
        order_field = query[order_match.start(1):order_match.end(1)]
        order_desc = order_match.group(2) == 'DESC'
        return order_field, order_desc
    
    return None, False

def parse_query(sql_query):
    """Parse the SQL-like query into its components."""
    if not sql_query or not isinstance(sql_query, str):
        raise ValueError("Query must be a non-empty string")
    
    query_upper = sql_query.upper()
    
    return {
        'select_fields': parse_select_clause(sql_query, query_upper),
        'where_conditions': parse_where_clause(sql_query, query_upper),
        'order_by': parse_order_by_clause(sql_query, query_upper)
    }

def apply_where_filter(dataset, conditions):
    """Apply WHERE conditions to filter the dataset."""
    ops_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    result = dataset
    
    for field, op, value in conditions:
        if op not in ops_map:
            raise ValueError(f"Unsupported operator: {op}")
        
        op_func = ops_map[op]
        filtered = []
        
        for record in result:
            if field not in record:
                continue
            try:
                if op_func(record[field], value):
                    filtered.append(record)
            except Exception:
                # Type comparison might fail, skip this record
                pass
        
        result = filtered
    
    return result

def apply_order_by(dataset, order_field, order_desc):
    """Apply ORDER BY to sort the dataset."""
    if not order_field:
        return dataset
    
    try:
        return sorted(dataset, key=lambda x: x.get(order_field, ''), reverse=order_desc)
    except Exception:
        raise ValueError(f"Cannot order by field: {order_field}")

def apply_select_projection(dataset, select_fields):
    """Apply SELECT projection to filter columns."""
    if select_fields is None:
        return dataset
    
    projected = []
    for record in dataset:
        new_record = {}
        for field in select_fields:
            if field in record:
                new_record[field] = record[field]
        projected.append(new_record)
    
    return projected

def execute_pipeline(dataset, parsed_query):
    """Execute the query pipeline on the dataset."""
    # Create a copy to avoid modifying the original
    result = dataset.copy()
    
    # Apply WHERE filtering
    result = apply_where_filter(result, parsed_query['where_conditions'])
    
    # Apply ORDER BY
    order_field, order_desc = parsed_query['order_by']
    result = apply_order_by(result, order_field, order_desc)
    
    # Apply SELECT projection
    result = apply_select_projection(result, parsed_query['select_fields'])
    
    return result

def execute_query_cmd(dataset_list, sql_query):
    """
    Execute a basic SQL-like statement on a dataset.
    
    Args:
        dataset_list: A list where each item is a dictionary representing data records
        sql_query: A string containing the SQL-like statement for execution
        
    Returns:
        A list of dictionaries, which are the results of the query operation
        
    Raises:
        ValueError: If there is an issue with the query format or when the query can't be processed successfully
    """
    parsed_query = parse_query(sql_query)
    return execute_pipeline(dataset_list, parsed_query)
