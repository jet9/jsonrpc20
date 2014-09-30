from jsonrpc20 import rpc_method

@rpc_method
def echo(msg):
    """test"""

    return msg

@rpc_method
def ping(msg="default_msg", unique="0"):
    """test"""

    return msg, unique

@rpc_method
def pong(pong_msg):
    """test"""

    print(pong_msg)

