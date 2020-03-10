import re
import sys
import subprocess

def extract_max_memory(archive_path):
    command = f"size -A -d {archive_path} | grep -o '.text[[:space:]]*[0123456789]*'"
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Handle errors
    if process.returncode != 0:
        if b'No such file' in process.stderr:
            raise FileNotFoundError()
        else:
            print("Error extracting archive information:")
            for line in process.stderr:
                sys.stdout.buffer.write(line)
            return []

    size = re.compile(b'(\d+)')

    max_memory_by_tile = []
    for line in process.stdout.split(b'\n'):
        match = size.search(line)
        if match:
            max_memory_by_tile.append(int(match.group()))

    return max_memory_by_tile
