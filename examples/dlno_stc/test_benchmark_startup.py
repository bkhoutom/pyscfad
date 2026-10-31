"""Check Linux worker stack defaults after the benchmark startup re-exec."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import unittest


WORKER = r'''
import ctypes, json, os, resource, sys, threading
sys.path.insert(0, sys.argv[1])
from benchmark_affinity import ensure_worker_stack
ensure_worker_stack()
assert sys.argv[2] == 'argument with spaces'
assert os.environ['BENCHMARK_STARTUP_TEST'] == 'preserved'
assert int(os.environ['BENCHMARK_STARTUP_PID']) == os.getpid()
libc = ctypes.CDLL(None)
libc.pthread_self.restype = ctypes.c_void_p
libc.pthread_getattr_np.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libc.pthread_attr_getstacksize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
libc.pthread_attr_destroy.argtypes = [ctypes.c_void_p]
result = {}
def check():
    # An aligned opaque allocation larger than glibc's pthread_attr_t.
    attr = (ctypes.c_long * 16)()
    size = ctypes.c_size_t()
    assert libc.pthread_getattr_np(libc.pthread_self(), attr) == 0
    assert libc.pthread_attr_getstacksize(attr, ctypes.byref(size)) == 0
    assert libc.pthread_attr_destroy(attr) == 0
    result['worker_stack_bytes'] = size.value
worker = threading.Thread(target=check)
worker.start()
worker.join()
result['stack_limit_bytes'] = resource.getrlimit(resource.RLIMIT_STACK)[0]
print(json.dumps(result))
'''


@unittest.skipUnless(sys.platform == 'linux', 'Linux worker stack startup')
class BenchmarkStartupTest(unittest.TestCase):
    def test_real_worker_stack_and_command_survive_reexec(self):
        for initial in (resource.RLIM_INFINITY, 2 * 1024**2, 64 * 1024**2):
            with self.subTest(initial=initial):
                def child_setup():
                    resource.setrlimit(resource.RLIMIT_STACK, (initial, resource.RLIM_INFINITY))
                # Record PID inside the first interpreter before invoking the
                # helper; re-exec must preserve it and the command arguments.
                code = "import os\nos.environ.setdefault('BENCHMARK_STARTUP_PID', str(os.getpid()))\n" + WORKER
                env = dict(os.environ, BENCHMARK_STARTUP_TEST='preserved')
                result = subprocess.run(
                    [sys.executable, '-c', code, str(Path(__file__).resolve().parent),
                     'argument with spaces'], env=env, preexec_fn=child_setup,
                    capture_output=True, text=True, check=True, timeout=20)
                info = json.loads(result.stdout.splitlines()[-1])
                expected = max(16 * 1024**2, initial)
                self.assertEqual(info['stack_limit_bytes'], expected)
                self.assertGreaterEqual(info['worker_stack_bytes'], expected)


if __name__ == '__main__':
    unittest.main()
