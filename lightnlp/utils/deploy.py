import socket
from ..base.module import Module


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


class App:
    def __init__(self):
        self.services_list = []

    def add_service(self, module: Module, route_path="", host="localhost", port=None, debug=False):
        pass
