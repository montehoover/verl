import re
import operator
from functools import partial

def extract_fields(data: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, each containing only the specified fields.

    Raises:
        ValueError: If a specified field is not found in a dictionary.
    """
    result = []
    for record in data:
        new_record = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record: {record}")
            new_record[field] = record[field]
        result.append(new_record)
    return result

# Global map for query operators
OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    'LIKE': lambda val, pattern: re.search(pattern, str(val), re.IGNORECASE) is not None
    # Note: LIKE pattern matching is basic here. SQL LIKE has % and _ wildcards.
    # This basic version uses regex search for the pattern string.
}

QUERY_PATTERN = re.compile(
    r"SELECT\s+(?P<fields>.*?)"
    r"(?:\s+WHERE\s+(?P<where_clause>.*?))?"
    r"(?:\s+ORDER BY\s+(?P<orderby_clause>.*?))?$",
    re.IGNORECASE | re.DOTALL
)

CONDITION_PATTERN = re.compile(
    r"\s*(?P<field>\w+)\s*(?P<operator>[<>=!]+|LIKE)\s*(?P<value>.*)\s*",
    re.IGNORECASE
)

def _parse_value(value_str: str):
    """
    Parses a string value from a query into a Python type (str, int, float).
    Removes quotes from strings.
    """
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str # Treat as string if not quoted and not numeric

def _parse_condition(condition_str: str) -> tuple | None:
    """
    Parses a single condition string (e.g., "age > 30") into (field, operator_func, value).
    Returns None if parsing fails.
    For simplicity, this version handles one condition. Complex WHERE (AND/OR) needs more parsing.
    """
    match = CONDITION_PATTERN.match(condition_str)
    if not match:
        raise ValueError(f"Invalid WHERE clause condition: {condition_str}")

    parts = match.groupdict()
    field = parts['field']
    op_symbol = parts['operator'].upper()
    raw_value = parts['value']

    if op_symbol not in OPERATORS:
        raise ValueError(f"Unsupported operator: {op_symbol}")

    op_func = OPERATORS[op_symbol]
    parsed_value = _parse_value(raw_value)

    return field, op_func, parsed_value

def execute_custom_query(data: list[dict], query: str) -> list[dict]:
    """
    Executes a SQL-like query (SELECT, WHERE, ORDER BY) on a list of dictionaries.

    Args:
        data: A list of dictionaries.
        query: A SQL-like query string.
               Example: "SELECT name, age WHERE age > 30 ORDER BY name DESC"

    Returns:
        A new list of dictionaries matching the query.

    Raises:
        ValueError: If the query is invalid or a field is not found.
    """
    if not data:
        return []

    query_match = QUERY_PATTERN.fullmatch(query.strip())
    if not query_match:
        raise ValueError(f"Invalid query syntax: {query}")

    query_parts = query_match.groupdict()
    
    # Process SELECT
    fields_str = query_parts['fields'].strip()
    if fields_str == '*':
        # Select all fields from the first record as a template
        # Assumes all records have similar structure; for robustness, could union all keys.
        select_fields = list(data[0].keys()) if data else []
    else:
        select_fields = [f.strip() for f in fields_str.split(',')]

    processed_data = list(data) # Start with a copy

    # Process WHERE
    where_clause_str = query_parts['where_clause']
    if where_clause_str:
        # This basic version assumes a single condition in the WHERE clause.
        # For multiple conditions (AND/OR), a more complex parser is needed.
        condition_field, condition_op, condition_value = _parse_condition(where_clause_str.strip())
        
        filtered_data = []
        for record in processed_data:
            if condition_field not in record:
                # Or skip record, or treat as non-match, depending on desired SQL NULL behavior
                raise ValueError(f"Field '{condition_field}' in WHERE clause not found in record: {record}")
            
            record_value = record[condition_field]
            
            # Attempt type coercion for comparison if types are mismatched (e.g. record_value is int, condition_value is str '10')
            # This is a simple coercion; more robust type handling might be needed.
            try:
                if isinstance(record_value, (int, float)) and isinstance(condition_value, str):
                    condition_value_coerced = type(record_value)(condition_value)
                elif isinstance(record_value, str) and isinstance(condition_value, (int, float)):
                     # Avoid coercing record string to number unless it's a LIKE op
                    condition_value_coerced = condition_value 
                else:
                    condition_value_coerced = condition_value
            except ValueError: # Coercion failed
                condition_value_coerced = condition_value


            if condition_op(record_value, condition_value_coerced):
                filtered_data.append(record)
        processed_data = filtered_data

    # Process ORDER BY
    orderby_clause_str = query_parts['orderby_clause']
    if orderby_clause_str:
        parts = orderby_clause_str.strip().split()
        orderby_field = parts[0]
        reverse_sort = False
        if len(parts) > 1 and parts[1].upper() == 'DESC':
            reverse_sort = True
        
        # Check if orderby_field exists in the (potential) first record to avoid runtime errors during sort
        # This check assumes consistent fields; more robust check would inspect select_fields
        # or ensure the field exists in all records before sorting.
        if processed_data and orderby_field not in processed_data[0] :
             # A field used in ORDER BY must be present in the records.
             # If it was not part of SELECT *, or explicitly selected, this could be an issue.
             # Standard SQL allows ordering by fields not in SELECT, but data must be available.
            raise ValueError(f"Field '{orderby_field}' in ORDER BY clause not found in data records.")

        processed_data.sort(key=lambda r: r.get(orderby_field), reverse=reverse_sort)

    # Apply SELECT (projection) as the final step
    final_result = []
    for record in processed_data:
        projected_record = {}
        for field in select_fields:
            if field not in record:
                raise ValueError(f"Field '{field}' in SELECT clause not found in record: {record}")
            projected_record[field] = record[field]
        final_result.append(projected_record)
        
    return final_result
