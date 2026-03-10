def update_dict_values(data, updates):
    return {**data, **updates}

def count_data_types(data):
    type_counts = {}
    for value in data.values():
        value_type = type(value)
        if value_type in type_counts:
            type_counts[value_type] += 1
        else:
            type_counts[value_type] = 1
    return type_counts
