import re
import operator
from functools import partial # Included as per problem description

# Global definition for supported SQL comparison operators
OPERATORS = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '=': operator.eq,
    '!=': operator.ne
}

def _parse_value(val_str):
    """
    Converts a string value from a SQL query literal to its Python type.
    Handles integers, floats, and quoted strings.
    """
    val_str = val_str.strip()
    if (val_str.startswith("'") and val_str.endswith("'")) or \
       (val_str.startswith('"') and val_str.endswith('"')):
        return val_str[1:-1]  # String literal (remove quotes)
    try:
        return int(val_str)  # Integer
    except ValueError:
        try:
            return float(val_str)  # Float
        except ValueError:
            # If it's not a recognized number and not quoted, it's an unsupported format
            # or could be an unquoted string literal, which we'll disallow for clarity.
            raise ValueError(f"Unsupported value format: '{val_str}'. String literals must be quoted. Numbers should be valid numerals.")

def run_sql_query(dataset, sql_query):
    """
    Processes a custom SQL-like query on data represented as a list of dictionaries.
    Handles SELECT, WHERE, and ORDER BY clauses.

    Args:
        dataset: A list of dictionaries, where each dictionary is a record.
        sql_query: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the query is malformed or execution fails.
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of dictionaries.")
    if dataset and not all(isinstance(row, dict) for row in dataset):
        raise ValueError("All items in the dataset must be dictionaries.")

    # Normalize query: remove multiple spaces, strip leading/trailing whitespace
    query = re.sub(r'\s+', ' ', sql_query).strip()

    # 1. Parse SELECT clause
    select_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Malformed query: SELECT clause not found or invalid. Expected 'SELECT fields FROM ...'.")
    
    select_fields_str = select_match.group(1).strip()
    if not select_fields_str:
        raise ValueError("Malformed SELECT clause: no fields specified.")

    select_all_fields = (select_fields_str == '*')
    if select_all_fields:
        parsed_select_fields = [] # Will be determined from data later
    else:
        parsed_select_fields = [f.strip() for f in select_fields_str.split(',')]
        if not parsed_select_fields or any(not f for f in parsed_select_fields):
            raise ValueError("Malformed SELECT clause: contains empty or invalid field names.")

    # 2. Parse FROM clause (must be 'FROM data')
    # Regex ensures 'FROM data' is followed by WHERE, ORDER BY, or end of query.
    from_clause_regex = r"FROM\s+data(?:\s+WHERE|\s+ORDER BY|$)"
    from_match = re.search(from_clause_regex, query, re.IGNORECASE)
    if not from_match:
        # Check if FROM is present but not 'FROM data' or structure is wrong
        if re.search(r"FROM\s+", query, re.IGNORECASE):
             raise ValueError("Malformed query: Must be 'FROM data'. Other table names are not supported.")
        else:
             raise ValueError("Malformed query: FROM clause not found or invalid.")


    current_data = list(dataset) # Work on a copy

    # 3. Parse and apply WHERE clause (optional)
    where_match = re.search(r"WHERE\s+(.+?)(?:\s+ORDER BY|$)", query, re.IGNORECASE)
    if where_match:
        where_clause_str = where_match.group(1).strip()
        # Simple condition: field operator value (e.g., "age > 25")
        condition_parts = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", where_clause_str)
        if not condition_parts:
            raise ValueError(f"Invalid WHERE clause format: '{where_clause_str}'. Expected 'field operator value'.")
        
        where_field, where_op_str, where_val_raw = condition_parts.groups()
        where_val_raw = where_val_raw.strip()

        if where_op_str not in OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: '{where_op_str}'.")
        
        op_func = OPERATORS[where_op_str]
        
        try:
            compare_val = _parse_value(where_val_raw)
        except ValueError as e: # Catch error from _parse_value
            raise ValueError(f"Error parsing value in WHERE clause ('{where_val_raw}'): {e}")

        filtered_data = []
        for row in current_data:
            if where_field not in row:
                # If field doesn't exist in a row, condition is effectively false for that row
                continue

            field_val = row[where_field]
            
            try:
                if op_func(field_val, compare_val):
                    filtered_data.append(row)
            except TypeError:
                raise ValueError(
                    f"Type mismatch in WHERE clause for field '{where_field}'. "
                    f"Cannot compare data value '{field_val}' (type {type(field_val).__name__}) "
                    f"with query value '{compare_val}' (type {type(compare_val).__name__}) "
                    f"using operator '{where_op_str}'."
                )
        current_data = filtered_data

    # 4. Parse and apply ORDER BY clause (optional)
    # Regex ensures ORDER BY is at the end of the query (after stripping spaces).
    orderby_match = re.search(r"ORDER BY\s+(\w+)(?:\s+(ASC|DESC))?$", query, re.IGNORECASE)
    if orderby_match:
        orderby_field = orderby_match.group(1).strip()
        orderby_direction_str = orderby_match.group(2) # This is None if not specified
        
        is_desc = False
        if orderby_direction_str and orderby_direction_str.upper() == 'DESC':
            is_desc = True
        
        if current_data: # Only try to sort if there's data
            # Check if orderby_field exists in the first record (as a sample)
            if orderby_field not in current_data[0]:
                # Check if it exists in any record, to be more robust for heterogeneous data
                if not any(orderby_field in row for row in current_data):
                     raise ValueError(f"ORDER BY field '{orderby_field}' not found in any data records.")
            
            try:
                # Using .get(orderby_field) makes sort robust to rows missing the key (they'll be grouped based on None)
                # However, Python 3 sort will raise TypeError if types are mixed (e.g. int and str) in the column
                current_data.sort(key=lambda x: x.get(orderby_field), reverse=is_desc)
            except TypeError:
                raise ValueError(f"Cannot sort on field '{orderby_field}' due to incompatible data types in that column (e.g., mixing numbers and strings).")
    
    # 5. Apply SELECT clause (Projection)
    if select_all_fields:
        if current_data: # If data exists after filtering/sorting
            parsed_select_fields = list(current_data[0].keys()) # Assume consistent keys
        elif dataset: # If current_data is empty but original dataset wasn't
            parsed_select_fields = list(dataset[0].keys()) # Use fields from original dataset
        else: # Original dataset was also empty
            parsed_select_fields = []

    result_data = []
    if not parsed_select_fields and current_data : # e.g. SELECT * from table with data, but parsed_select_fields somehow not set
        # This case should ideally not be hit if select_all_fields logic is correct.
        # If parsed_select_fields is empty (e.g. SELECT * from empty table), result should be [].
         pass


    for row in current_data:
        selected_row = {}
        for field_name in parsed_select_fields:
            if field_name not in row:
                # This implies a specific field was selected that doesn't exist in this row.
                # Standard SQL might return NULL; here we raise an error for strictness.
                raise ValueError(f"Selected field '{field_name}' not found in data record: {row}. Ensure all selected fields exist in all rows after filtering.")
            selected_row[field_name] = row[field_name]
        result_data.append(selected_row)
        
    return result_data

```
