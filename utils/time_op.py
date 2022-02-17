import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

import time

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class MeasureTime(object):
    def __init__(self, key, print_option=True):
        self._key = key
        self.print_option=print_option

    def __enter__(self):
        self._time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.print_option:
            print(Color.GREEN+'Elapsed time {}: {}'.format(self._key, time.time() - self._time)+Color.GREEN)

    def get_time(self):
        return time.time()-self._time

