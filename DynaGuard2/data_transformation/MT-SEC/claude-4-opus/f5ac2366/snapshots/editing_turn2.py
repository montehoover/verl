def construct_internal_resource_url(base_url, resource_id):
    return base_url + resource_id

def construct_categorized_resource_url(base_url, resource_id, resource_type):
    valid_types = ['document', 'video', 'image', 'audio', 'data']
    if resource_type not in valid_types:
        raise ValueError(f"Invalid resource type. Must be one of: {', '.join(valid_types)}")
    return base_url + resource_type + '/' + resource_id
