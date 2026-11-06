"""Large cloud artifacts must not be buffered after their client disconnects."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from webgui.server import Handler

class CloudFile:
    name = 'large-model.safetensors'
    suffix = '.safetensors'
    def __init__(self, size=20_000_007, actual=None):
        self.size = size
        self.actual = size if actual is None else actual
        self.position = 0
        self.max_read = 0
        self.opened = False
        self.closed = False
        self.before_read = lambda: None
    def is_file(self): return True
    def stat(self): return SimpleNamespace(st_size=self.size, st_mtime_ns=123)
    def read_bytes(self): raise AssertionError('Whole cloud file was buffered')
    def open(self, mode):
        assert mode == 'rb'
        self.opened = True
        return self
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def read(self, size):
        self.before_read()
        self.max_read = max(self.max_read, size)
        count = min(size, self.actual - self.position)
        self.position += count
        return b'x' * count

class LargeFileStreamingTests(unittest.TestCase):
    def handler(self, *, disconnect=False, headers=None):
        handler = Handler.__new__(Handler)
        handler.headers = headers or {}
        handler.close_connection = False
        handler.status = None
        handler.sent_headers = {}
        handler.headers_finished = False
        handler.bytes_written = 0
        handler.send_response = lambda status: setattr(handler, 'status', status)
        handler.send_header = lambda name, value: handler.sent_headers.update({name: value})
        handler.end_headers = lambda: setattr(handler, 'headers_finished', True)
        handler.log_error = Mock()
        def write(data):
            if disconnect: raise ConnectionResetError()
            handler.bytes_written += len(data)
        handler.wfile = SimpleNamespace(write=write)
        return handler
    def test_headers_precede_cloud_reads_and_memory_stays_bounded(self):
        file = CloudFile()
        handler = self.handler()
        file.before_read = lambda: self.assertTrue(handler.headers_finished)
        handler._file(file, cache='private, no-cache')
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.sent_headers['Content-Length'], str(file.size))
        self.assertEqual(handler.bytes_written, file.size)
        self.assertLessEqual(file.max_read, 64 * 1024)
        self.assertTrue(file.closed)
    def test_disconnected_client_stops_cloud_read_after_first_chunk(self):
        file = CloudFile()
        handler = self.handler(disconnect=True)
        handler._file(file)
        self.assertEqual(file.position, 64 * 1024)
        self.assertTrue(file.closed)
        self.assertTrue(handler.close_connection)
    def test_conditional_hit_never_opens_cloud_file(self):
        file = CloudFile()
        handler = self.handler(headers={'If-None-Match': f'W/"7b-{file.size:x}"'})
        handler._file(file)
        self.assertEqual(handler.status, 304)
        self.assertFalse(file.opened)
    def test_growing_file_does_not_exceed_declared_content_length(self):
        file = CloudFile(actual=21_000_000)
        handler = self.handler()
        handler._file(file)
        self.assertEqual(handler.bytes_written, file.size)
        self.assertEqual(file.position, file.size)
    def test_truncated_stream_cannot_reuse_connection(self):
        file = CloudFile(actual=100)
        handler = self.handler()
        handler._file(file)
        self.assertEqual(handler.bytes_written, 100)
        self.assertTrue(handler.close_connection)
        self.assertTrue(file.closed)

if __name__ == '__main__': unittest.main()
