import time
import sys

print("Script starting...", flush=True)
print(f"Python version: {sys.version}", flush=True)

try:
    counter = 1
    while True:
        print(f"Hello World {counter}", flush=True)
        sys.stdout.flush()  # Extra flush
        counter += 1
        time.sleep(2)
except Exception as e:
    print(f"Error occurred: {e}", flush=True)
    sys.stdout.flush()  # Extra flush
