import os.path
import unittest
import json

from jsonrpc20 import rpc_method, process_request

VALID_REQUEST01 = '{"jsonrpc": "2.0", "method": "echo", "params": ["test echo"], "id": 1}'
INVALID_REQUEST01 = '{"jsonrpc": "2.0", "method": "echo", "params": [1, "test echo"], "id": 1}'
INVALID_REQUEST02 = '{"jsonrpc": "2.0", "method": "echo2", "params": ["test echo"], "id": 1}'
INVALID_REQUEST03 = '{"jsonrpc": "2.0", "method": "echo", "params": ["test echo"], "id": 1]'
INVALID_REQUEST04 = '{"method": "echo", "params": ["test echo"], "id": 1}'

@rpc_method
def echo(msg):
    """Test echo"""

    return msg

class JsonRpcTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.__test_dir = os.path.dirname(__file__)
        self.do_teardown = True

    def setUp(self):
        """Init tests"""

        self.do_teardown = True

    def tearDown(self):
        """Cleanup tests"""

        if self.do_teardown:
            pass

    def test_010_process_requests(self):
        """010 Test process_request"""

        res = json.loads(process_request("test-jsonrpc", VALID_REQUEST01))
        self.assertEqual(res["result"], "test echo")

        res = json.loads(process_request("test-jsonrpc", INVALID_REQUEST01))
        self.assertEqual(res["error"]["code"], -32602)

        res = json.loads(process_request("test-jsonrpc", INVALID_REQUEST02))
        self.assertEqual(res["error"]["code"], -32601)

        res = json.loads(process_request("test-jsonrpc", INVALID_REQUEST03))
        self.assertEqual(res["error"]["code"], -32700)

        res = json.loads(process_request("test-jsonrpc", INVALID_REQUEST04))
        self.assertEqual(res["error"]["code"], -32600)


if __name__ == "__main__":
    unittest.main()
