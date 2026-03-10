import json


def build_url_with_query(base_url: str, path: str, query_params: dict) -> None:
    """
    Log the provided base_url, path, and query_params to 'url_log.txt' as a single plain-text line.
    The entry is appended to the file. The function does not return anything.
    """
    qp_serialized = json.dumps(query_params if query_params is not None else {}, ensure_ascii=False, sort_keys=True)
    log_line = f"base_url={base_url} | path={path} | query_params={qp_serialized}\n"
    with open("url_log.txt", "a", encoding="utf-8") as f:
        f.write(log_line)
