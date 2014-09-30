from __future__ import print_function

import os
import logging
import json
import re

from jsonschema import validate, ValidationError
from wsgiref.simple_server import make_server
from ndict import NDict

# standart specified errors
RPC_PARSE_ERROR = (-32700, "Parse error")
INVALID_REQUEST = (-32600, "Invalid Request")
METHOD_NOT_FOUND = (-32601, "Method not found")
INVALID_PARAMS = (-32602, "Invalid params")
INTERNAL_ERROR = (-32603, "Internal error")

# custom erros
MODULE_NOT_FOUND = (-32000, "Module not found")

# jsonrpc 2.0 validating schema
REQUEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "A JSON RPC 2.0 request",
    "oneOf": [
        {
            "description": "An individual request",
            "$ref": "#/definitions/request"
        },
        {
            "description": "An array of requests",
            "type": "array",
            "items": {"$ref": "#/definitions/request"}
        }
    ],
    "definitions": {
        "request": {
            "type": "object",
            "required": ["jsonrpc", "method"],
            "properties": {
                "jsonrpc": {"enum": ["2.0"]},
                "method": {
                    "type": "string"
                },
                "id": {
                    "type": ["string", "number", "null"],
                    "note": [
                        "While allowed, null should be avoided: http://www.jsonrpc.org/specification#id1",
                        "While allowed, a number with a fractional part should be avoided: http://www.jsonrpc.org/specification#id2"
                    ]
                },
                "params": {
                    "type": ["array", "object"]
                }
            }
        }
    }
}
RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "A JSON RPC 2.0 response",
    "oneOf": [
        {"$ref": "#/definitions/success"},
        {"$ref": "#/definitions/error"},
        {
            "type": "array",
            "items": {
                "oneOf": [
                    {"$ref": "#/definitions/success"},
                    {"$ref": "#/definitions/error"}
                ]
            }
        }
    ],
    "definitions": {
        "common": {
            "required": ["id", "jsonrpc"],
            "not": {
                "description": "cannot have result and error at the same time",
                "required": ["result", "error"]
            },
            "type": "object",
            "properties": {
                "id": {
                    "type": ["string", "integer", "null"],
                    "note": [
                        "spec says a number which should not contain a fractional part",
                        "We choose integer here, but this is unenforceable with some languages"
                    ]
                },
                "jsonrpc": {"enum": ["2.0"]}
            }
        },
        "success": {
            "description": "A success. The result member is then required and can be anything.",
            "allOf": [
                {"$ref": "#/definitions/common"},
                {"required": ["result"]}
            ]
        },
        "error": {
            "allOf": [
                {"$ref": "#/definitions/common"},
                {
                    "required": ["error"],
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["code", "message"],
                            "properties": {
                                "code": {
                                    "type": "integer",
                                    "note": ["unenforceable in some languages"]
                                },
                                "message": {"type": "string"},
                                "data": {
                                    "description": "optional, can be anything"
                                }
                            }
                        }
                    }
                }
            ]
        }
    }
}

__RPC = {}

LOG = logging.getLogger(__name__)

__all__ = ["process_request", "wsgi_application", "start_standalone_server"]

__version__ = "0.1a"

class BaseError(Exception):
    def __init__(self, msg):
        LOG.exception("Exception raised")


class MethodNotFound(BaseError):
    pass


class ModuleNotFound(BaseError):
    pass


def _get_rpc(module, method):
    """Get filled RCP struct with method and corrosponding functions"""

    if module not in __RPC.keys():
        raise ModuleNotFound(module)
    try:
        return __RPC[module][method]
    except:
        raise MethodNotFound(method)


def _register_rpc_method(function):
    """Register rpc method"""

    modname = os.path.splitext(os.path.basename(function.func_code.co_filename))[0]

    if modname not in __RPC.keys():
        __RPC[modname] = {}

    __RPC[modname][function.func_name] = function


def _validate_request(request):
    """Validate json request

    :request    dict or json_sring"""

    if isinstance(request, dict):
        validate(request, REQUEST_SCHEMA)

    elif isinstance(request, basestring):
        validate(json.loads(request), REQUEST_SCHEMA)

    else:
        raise TypeError("Unknown request type: {0}".format(type(request)))


def _validate_response(response):
    """Validate json response

    :request    dict or json_sring"""

    if isinstance(response, dict):
        validate(response, RESPONSE_SCHEMA)

    elif isinstance(response, basestring):
        validate(json.loads(response), RESPONSE_SCHEMA)

    else:
        raise TypeError("Unknown response type: {0}".format(type(response)))


def _extract_module_name(path):
    """Extract module name from url path"""

    p = re.compile("^/([a-zA-Z_][a-zA-Z0-9_]*)/json[/]*$")
    matches = p.match(path)

    if matches is not None:
        return matches.group(1)

    return None


def _parse_request(environ):
    """Parse request and get common parameters"""

    request = NDict({
        "method": environ.get("REQUEST_METHOD", "GET"),
        "query_string:": environ.get("QUERY_STRING", ""),
        "path_info": environ.get("PATH_INFO", "/"),
        "content_length": int(environ.get("CONTENT_LENGTH", 0))
    })

    request.body = environ["wsgi.input"].read(request.content_length)
    request.module = _extract_module_name(request.path_info)
    LOG.info("Module name: {0}".format(request.module))

    return request


def rpc_method(function):
    """Decorator: make function possible to RPC"""

    _register_rpc_method(function)

    def wrapper(*args, **kwargs):
        function(*args, **kwargs)

    return wrapper


def json_ok(result, _id):
    """Get ok json message"""

    return json.dumps({
        "jsonrpc": "2.0",
        "result": result,
        "id": _id
    }, indent=True)


def json_error(code, message, _id=None, data=None):
    """Get error json message"""

    return json.dumps({
        "jsonrpc": "2.0",
        "error": {
            "code": code,
            "message": message,
            "data": data
        },
        "id": _id
    }, indent=True)


def process_request(module, request):
    """Process RPC request"""

    try:
        try:
            request = json.loads(request)
        except Exception as e:
            return json_error(RPC_PARSE_ERROR[0], RPC_PARSE_ERROR[1], data=str(e))

        try:
            _validate_request(request)
        except ValidationError as e:
            return json_error(INVALID_REQUEST[0], INVALID_REQUEST[1])

        method = request["method"]
        params = request["params"]
        request_id = request.get("id", None)

        try:
            function = _get_rpc(module, method)
        except MethodNotFound as e:
            return json_error(*METHOD_NOT_FOUND, _id=request_id, data=str(e))
        except ModuleNotFound as e:
            return json_error(*MODULE_NOT_FOUND, _id=request_id, data=str(e))

        try:
            if isinstance(params, list):
                result = function(*params)
            elif isinstance(params, dict):
                result = function(**params)
            else:
                raise TypeError("Invalid parameter type: {0}".format(type(params)))
        except TypeError as e:
            return json_error(*INVALID_PARAMS, _id=request_id, data=str(e))

        if request_id is not None:
            return json_ok(result, request_id)

    except Exception as e:
        return json_error(*INTERNAL_ERROR, data=str(e))


def wsgi_application(environ, start_response):
    """WSGI Application for using in web servers"""

    if environ["REQUEST_METHOD"] != "POST":
        status = "405 Method Not Allowed"
        headers = []
        start_response(status, headers)
        return [status]

    request = _parse_request(environ)
    result = process_request(request.module, request.body)

    if result is None:
        status = "200 OK"
        start_response(status, [])
        return []

    status = "200 OK"
    headers = [('Content-type', 'application/json')]
    start_response(status, headers)

    #return ["{0}: {1}\n".format(k, environ[k]) for k in sorted(environ)]
    return [result]


def start_standalone_server(address="localhost", port=8000, app=wsgi_application):
    """Start standalone http server for processing requests"""

    httpd = make_server(address, port, app)
    LOG.info("Serving on  {0}:{1}...".format(address, port))
    httpd.serve_forever()


if __name__ == "__main__":

    # simple example

    req = '{"jsonrpc": "2.0", "method": "echo", "params": [1, "test echo"], "id": 1}'

    print(process_request("defmod", req))
