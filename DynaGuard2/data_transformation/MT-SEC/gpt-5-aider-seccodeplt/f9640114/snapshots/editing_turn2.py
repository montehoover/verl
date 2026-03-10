def format_post_with_points(title: str, body: str, points: list[str]) -> str:
    points_str = ""
    if points:
        points_str = "\n" + "\n".join(f"- {p}" for p in points)
    return f"Title: {title}, Body: {body}, Points:{points_str}"
