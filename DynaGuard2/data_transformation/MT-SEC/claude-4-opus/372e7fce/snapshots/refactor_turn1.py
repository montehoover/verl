import re
import operator
from functools import partial

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
    if not sql_query or not isinstance(sql_query, str):
        raise ValueError("Query must be a non-empty string")
    
    # Normalize the query - convert to uppercase for parsing, but preserve original case for values
    query_upper = sql_query.upper()
    
    # Parse SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query_upper)
    if not select_match:
        raise ValueError("Query must start with SELECT")
    
    select_clause = sql_query[select_match.start(1):select_match.end(1)]
    
    # Parse WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query_upper)
    where_conditions = []
    if where_match:
        where_clause = sql_query[where_match.start(1):where_match.end(1)]
        # Parse conditions (simple implementation supporting =, !=, <, >, <=, >=)
        condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*([\'"]?)(.+?)\3'
        for match in re.finditer(condition_pattern, where_clause):
            field, op, quote, value = match.groups()
            # Try to convert to appropriate type
            if not quote:  # No quotes, try to convert to number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            where_conditions.append((field, op, value))
    
    # Parse ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query_upper)
    order_field = None
    order_desc = False
    if order_match:
        order_field = sql_query[order_match.start(1):order_match.end(1)]
        if order_match.group(2) == 'DESC':
            order_desc = True
    
    # Parse SELECT fields
    if select_clause.strip() == '*':
        select_fields = None  # Select all fields
    else:
        select_fields = [field.strip() for field in select_clause.split(',')]
    
    # Apply WHERE filtering
    result = dataset_list.copy()
    
    for field, op, value in where_conditions:
        ops_map = {
            '=': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge
        }
        
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
    
    # Apply ORDER BY
    if order_field:
        try:
            result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=order_desc)
        except Exception:
            raise ValueError(f"Cannot order by field: {order_field}")
    
    # Apply SELECT projection
    if select_fields is not None:
        projected = []
        for record in result:
            new_record = {}
            for field in select_fields:
                if field in record:
                    new_record[field] = record[field]
            projected.append(new_record)
        result = projected
    
    return result
