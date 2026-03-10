def load_serialized_data(filepath: str, data_format: str):
    allowed_serialized_formats = ["json", "text"]
    if data_format not in allowed_serialized_formats:
        raise ValueError(f"Unsupported or untrusted data format: {data_format}")

    if data_format == "json":
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif data_format == "text":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
