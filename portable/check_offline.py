"""Acceptance prerequisite: verify that the bundled interpreter cannot connect."""
import socket

try:
    with socket.create_connection(('1.1.1.1', 443), timeout=4):
        pass
except OSError as error:
    print('Network connection denied/unavailable:', type(error).__name__)
else:
    raise SystemExit('The bundled interpreter still has network access; offline acceptance is invalid.')
