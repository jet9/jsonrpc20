#!/bin/sh

curl -XPOST http://localhost:9000/defmod/json --data '{"jsonrpc": "2.0", "params": ["XXX"], "method": "ping", "id": "asdfa" }'
