import os
import sys


def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        print(f'[ERROR] {filename} not found')
        sys.exit()


