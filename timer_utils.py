# Adapted from https://realpython.com/python-timer/
import time
import numpy as np

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"---\nElapsed time: {elapsed_time:0.10f} seconds\n---")


if __name__== "__main__":
    t = Timer()
    t.start()
    a = np.ones([9, 5, 7, 4])
    c = np.ones([9, 5, 4, 3])
    res = np.dot(a, c)
    t.stop()