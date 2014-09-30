from jsonrpc20 import rpc_method

@rpc_method
def echo(msg):
    """test"""

    print(msg)

@rpc_method
def ping(ping_msg):
    """test"""

    print(ping_msg)

@rpc_method
def pong(pong_msg):
    """test"""

    print(pong_msg)

