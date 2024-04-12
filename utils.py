import os


def get_proc_filename(filename: str) -> str:
    parts = filename.rsplit('.', 1)

    if len(parts) <= 1:
        return None

    new_filename = parts[0] + '-proc.' + parts[1]

    return new_filename


def add_prefix_filename(file_path: str, prefix: str) -> str:
    directory, filename = os.path.split(file_path)
    new_filename = prefix + filename

    return os.path.join(directory, new_filename)


def calculate_time(total_seconds):
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    return hours, minutes, seconds
