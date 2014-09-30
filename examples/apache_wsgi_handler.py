#!/usr/bin/env python

import sys
import os.path

sys.path.append(os.path.dirname(__file__))

# import custom modules with implemented methods
import defmod
import defmod2

from jsonrpc20 import wsgi_application

application = wsgi_application
"""Main entry point of the WSGI application."""

