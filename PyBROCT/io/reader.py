from .fields import FixedFieldsParser
from .format.broct import BroctFormat
from .format.tiny_broct import TinyBroctFormat

import logging # pylint:disable=wrong-import-order
logger = logging.getLogger(__name__)

FORMAT_MAP = {
    70: BroctFormat,
    83: BroctFormat,
    116: TinyBroctFormat,
}

class ScanReader:
    def __init__(self):
        self._meta_parser = FixedFieldsParser([('meta', 'i')])

        self._file = None
        self._header = None

        self._format = None

        self._count = 0
        self._position = None
        self._start = None
        self._size = None

    def open(self, obj):
        self.close()

        if isinstance(obj, str):
            self._file = open(obj, 'rb')
        else:
            self._file = obj

        self._header = {}
        self._meta_parser.read(self._file, self._header)

        meta = self._header.get('meta')
        try:
            self._format = FORMAT_MAP[meta]()
        except KeyError:
            raise RuntimeError(f'unsupported meta type {meta}')

        self._format.read_header(self._file, self._header)

        # figure out number of volumes
        self._size = self._format.size_volume(self._header)
        self._start = self._file.tell()
        self._file.seek(0, 2)
        self._count = (self._file.tell() - self._start) // self._size

        # return to start
        self.seek_abs(0)

    def close(self):
        if not self._file:
            return

        self._file.close()
        self._file = None

        self._header = None
        self._count = 0
        self._position = 0

    def seek_abs(self, idx):
        if idx < 0:
            idx += self.count

        if idx < 0:
            raise RuntimeError(f'seek before start: {idx}')
        elif idx >= self.count:
            raise RuntimeError(f'seek past end: {idx} >= {self.count}')

        self._file.seek(self._start + idx * self._size, 0)
        self._position = idx

    def seek_rel(self, offset):
        self.seek_abs(self.position + offset)

    def read(self):
        result = self.header.copy()
        self._format.read_volume(self._file, result)
        return result

    @property
    def shape(self):
        return self._format.shape_volume(self.header)

    @property
    def header(self):
        return self._header

    @property
    def count(self):
        return self._count

    @property
    def position(self):
        return self._position

def scans(f, skip=None, count=None, step=None):
    reader = ScanReader()
    reader.open(f)

    idx = skip or 0
    if idx < 0:
        idx += reader.count

    n = 0
    while True:
        if count and n >= count:
            break

        if idx < 0 or idx >= reader.count:
            break

        reader.seek_abs(idx)
        yield (idx, reader.read())

        idx += step or 1
        n += 1

    reader.close()
