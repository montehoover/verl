from html import escape


def format_user_info(name: str, bio: str) -> str:
    return f"<div><h2>{escape(str(name))}</h2><p>{escape(str(bio))}</p></div>"


def generate_labels_html(labels: list[str]) -> str:
    spans = []
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        spans.append(f"<span>{escape(text)}</span>")
    return "".join(spans)
