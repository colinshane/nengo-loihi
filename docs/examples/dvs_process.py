import nengo
import numpy as np

class DVSFileProcess:
    def __init__(self, filename):
        self.filename = filename
        self.width = 240
        self.height = 180
        self.polarity = 2

    def read_data(self):
        with open(self.filename, 'rb') as fh:
            header = True
            buf = b''
            while header:
                buf = buf + f.read(1024)
                while buf[0] == b'#':
                    end = buf.
                    buf = buf

                if buf[0]
                line = f.readline()
                if not line.startswith('#'):
                    break
