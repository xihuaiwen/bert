import os
import re
import shutil

def copy(src, dst):
    if src != dst:
        shutil.copy(src, dst)

OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def finish(string):
    print(f"{OKGREEN}[Complete]{ENDC} {string}")
def warn(string):
    print(f"{WARNING}[Warning]{ENDC} {string}")
def fail(string):
    print(f"{FAIL}[Error]{ENDC} {string}")

def detect_framework(stderr_file, popart_log_file):
    # Popart
    if os.path.isfile(popart_log_file):
        return "popart"
    # Tensorflow
    tf = re.compile('.*StreamExecutor device \(\d\): Poplar.*')
    with open(stderr_file, 'r') as f:
        for line in f:
            if tf.match(line):
                return "tensorflow"
    return "unknown"

