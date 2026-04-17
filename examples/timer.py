import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_ping_time = self.start_time
        self.elapsed_time = 0

    def ping(self, message="Time since last ping: "):
        current_time = time.time()
        elapsed_since_last_ping = current_time - self.last_ping_time
        self.last_ping_time = current_time
        self.elapsed_time = current_time - self.start_time
        if message is not None:
            print(f"{message} {elapsed_since_last_ping:.2f} seconds")

    def total(self, message="Total time elapsed: "):
        total_elapsed_time = time.time() - self.start_time
        print(f"{message} {total_elapsed_time:.2f} seconds")