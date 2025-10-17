import os
import sys
from datetime import datetime

class Logger(object):
    def __init__(self, stream=sys.stdout, log_dir=None):
        self.log = None
        self.terminal = stream

        try:
            self.log_dir = log_dir or os.path.dirname(os.path.abspath(__file__))
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: Failed to initialize log directory: {e}", file=sys.stderr)
            self.log_dir = os.getcwd()

        try:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S") + f"-{now.microsecond // 1000:03d}"
            self.log_path = os.path.join(self.log_dir, f"{timestamp}.log")
            self.log = open(self.log_path, 'a+', buffering=1)
        except Exception as e:
            print(f"Critical Error: Cannot open log file: {e}", file=sys.stderr)
            self.log = None

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()

    def __del__(self):
        if self.log:
            self.log.close()