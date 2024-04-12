def get_proc_filename(filename: str) -> str:
    parts = filename.rsplit('.', 1)

    if len(parts) <= 1:
        return None

    new_filename = parts[0] + '-proc.' + parts[1]

    return new_filename

def remove_extension(filename: str) -> str:
    parts = filename.rsplit('.', 1)

    if len(parts) <= 1:
        return None

    return parts[0]