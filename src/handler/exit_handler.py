import threading
import signal

EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Program termination signalled. Exiting cleanly!", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
signal.signal(signal.SIGUSR2, _clean_exit_handler)
