import re
import operator
from functools import partial

def run_custom_query(dataset, query):
    """
    Executes a basic SQL-like statement on a dataset.

    Args:
        dataset: A list of dictionaries representing data records.
        query: A string containing the SQL-like statement.

    Returns:
        A list of dictionaries, results of the query.

    Raises:
        ValueError: If query format is invalid or processing fails.
    """

    OPS = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '=': operator.eq,  # Common SQL alias for ==
    }

    # Regex to parse the main SQL query structure
    query_regex = re.compile(
        r"SELECT\s+(?P<fields>.+?)\s+"
        r"FROM\s+(?P<tablename>\w+)"
        r"(?:\s+WHERE\s+(?P<where_clause>.+?))?"
        r"(?:\s+ORDER BY\s+(?P<orderby_field>\w+)(?:\s+(?P<orderby_direction>ASC|DESC))?)?$",
        re.IGNORECASE  # Make query keywords case-insensitive
    )

    match = query_regex.match(query.strip())
    if not match:
        raise ValueError("Invalid query format")

    query_parts = match.groupdict()

    # Validate table name; problem implies 'data' as the source
    if query_parts['tablename'].lower() != 'data':
        raise ValueError(f"Query must be FROM 'data'; found '{query_parts['tablename']}'")

    current_data = list(dataset)  # Work on a copy to avoid modifying the original

    # 1. Process WHERE clause (filter data)
    if query_parts['where_clause']:
        where_str = query_parts['where_clause']
        # Regex to parse "field operator value" from WHERE clause
        where_regex = re.compile(r"^\s*(\w+)\s*([><=!]+)\s*(.+)\s*$", re.IGNORECASE)
        where_match = where_regex.match(where_str)
        
        if not where_match:
            raise ValueError(f"Invalid WHERE clause format: {where_str}")

        where_field, op_str, val_str = where_match.groups()
        
        if op_str not in OPS:
            raise ValueError(f"Unsupported operator in WHERE clause: {op_str}")
        op_func = OPS[op_str]

        # Parse the value in the WHERE clause (string, int, float)
        parsed_val = None
        val_str_stripped = val_str.strip()

        if (val_str_stripped.startswith("'") and val_str_stripped.endswith("'")) or \
           (val_str_stripped.startswith('"') and val_str_stripped.endswith('"')):
            parsed_val = val_str_stripped[1:-1]  # String value (remove quotes)
        else:
            try:
                parsed_val = int(val_str_stripped)  # Integer value
            except ValueError:
                try:
                    parsed_val = float(val_str_stripped)  # Float value
                except ValueError:
                    # Value is not a recognized number and not a quoted string
                    raise ValueError(
                        f"WHERE clause value '{val_str_stripped}' must be a number or a quoted string."
                    )
        
        if not current_data: # No data to filter
            current_data = []
        else:
            # Assuming homogeneous data, check field existence using the first record
            if where_field not in current_data[0]:
                 raise ValueError(f"Field '{where_field}' not found in dataset for WHERE clause.")

            filtered_data = []
            for row in current_data:
                row_val = row.get(where_field)

                if row_val is None:  # In SQL, NULL compared to anything is usually UNKNOWN (effectively false in WHERE)
                    continue
                
                try:
                    # Attempt to cast row_val to the type of parsed_val for comparison
                    # e.g., if parsed_val is int(25), try to convert row_val to int before comparing
                    actual_row_val = type(parsed_val)(row_val) 
                    if op_func(actual_row_val, parsed_val):
                        filtered_data.append(row)
                except (ValueError, TypeError):
                    # Skip row if types are incompatible for comparison or conversion failed
                    # e.g., trying to convert "abc" to int, or comparing "text" > 10
                    pass 
            current_data = filtered_data

    # 2. Process ORDER BY clause (sort data)
    if query_parts['orderby_field']:
        orderby_field = query_parts['orderby_field']
        
        if current_data:  # Only sort if there's data
            # Check if the orderby_field exists in the data (assuming homogeneity)
            if orderby_field not in current_data[0]:
                raise ValueError(f"Field '{orderby_field}' not found in dataset for ORDER BY clause.")
            
            try:
                # Sort key: (is_None, value) ensures Nones are grouped and sorted predictably.
                # (True, ...) comes after (False, ...), so Nones are last in ASC.
                current_data.sort(
                    key=lambda x: (x.get(orderby_field) is None, x.get(orderby_field)),
                    reverse=(query_parts['orderby_direction'] is not None and 
                             query_parts['orderby_direction'].upper() == 'DESC')
                )
            except TypeError:
                # This can happen if 'orderby_field' contains mixed, non-comparable types (e.g., strings and numbers)
                raise ValueError(
                    f"Field '{orderby_field}' contains mixed non-comparable types; cannot sort."
                )

    # 3. Process SELECT clause (project data)
    fields_str = query_parts['fields'].strip()
    
    selected_fields = []
    if fields_str == '*':
        if not current_data:
            return []  # SELECT * from an empty (possibly filtered) dataset is an empty list
        # Assuming homogeneous data, get all field names from the first record
        selected_fields = list(current_data[0].keys())
    else:
        selected_fields = [f.strip() for f in fields_str.split(',')]
        # Check for invalid field names (e.g., empty string from "f1,,f2")
        if not all(selected_fields): 
            raise ValueError("Invalid field list in SELECT clause: contains empty or invalid field names.")

        if current_data:  # Validate field names if data exists
            first_record_keys = current_data[0].keys()
            for sf in selected_fields:
                if sf not in first_record_keys:
                    raise ValueError(f"Field '{sf}' not found in dataset for SELECT clause.")
        # If current_data is empty (e.g., WHERE clause filtered everything out),
        # the result will be an empty list, which is correct. No validation error needed here.
    
    # Construct the final result set
    result = []
    for row in current_data:
        result_row = {field: row.get(field) for field in selected_fields}
        result.append(result_row)

    return result
