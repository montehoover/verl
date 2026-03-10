import json

def build_url_with_query(base_url: str, path: str, query_params: dict) -> None:
    """
    Log the base URL, path, and query parameters to a file named 'url_log.txt'.
    Writes a single plain-text log entry and returns nothing.
    """
    entry = f"base_url={base_url} path={path} query_params={json.dumps(query_params, ensure_ascii=False, sort_keys=True)}\n"
    with open("url_log.txt", "a", encoding="utf-8") as f:
        f.write(entry)
