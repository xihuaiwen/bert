import os
import re
import glob
from functools import reduce
from .utils import copy, warn
import subprocess

def extract_tf_tile_mapping(log_file, pretty=False):
    # Use regex to get mappings line from log text file:
    start = re.compile('.*Dumping tensor mapping.*')
    mappings = re.compile('^.*{"mappings"')
    count = 0
    json = []
    opened = False
    with open(log_file, 'r') as f:
        for line in f:
            if opened:
                m = mappings.match(line)
                if m:
                    opened = False
                    json.append('{"mappings"' + line.split('{"mappings"', 1)[1])
            else:
                m = start.match(line)
                if m:
                    opened = True
                    count += 1

    if count == 0:
        warn(f"Could not reliably extract mappings from log ({count} matches): has the format changed?")
        return False

    # String join. Maybe stripping leading whitespace.
    return json[-1]

def largest_tile_mapping_file(tile_mapping_dir):
    path = os.path.join(tile_mapping_dir, "*tensor_map.json")
    files = glob.glob(path)
    if len(files) < 1:
        return None
    idx,__ = max(enumerate(map(lambda dot_file: os.stat(dot_file).st_size, files)), key=lambda file: file[1])
    return files[idx]

def convert_graph_to_svg(dot_file, output_file=None):
    result = subprocess.run(['dot', dot_file, '-Tsvg', '-O'], timeout=60)
    if result.returncode != 0:
        raise Exception("Converting dot to svg failed")
    if output_file:
        copy(f"{dot_file}.svg", output_file)

def find_xla_dot_file(xla_dump_dir):
    path = os.path.join(xla_dump_dir, "*after_forward-allocation*.dot")
    files = glob.glob(path)
    if len(files) < 1:
        return None
    idx,__ = max(enumerate(map(lambda dot_file: os.stat(dot_file).st_size, files)), key=lambda file: file[1])
    return files[idx]
