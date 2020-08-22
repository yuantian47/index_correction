import numpy

from PyBROCT.io.fields import FixedFieldsParser, VariableStringParser

def _mul(l):
    p = 1
    for o in l:
        p *= o
    return p

class TinyBroctFormat:
    def __init__(self):
        self._parsers = [
            VariableStringParser('notes', 'Q'),
            FixedFieldsParser([
                ('depth_samples',    'Q'),
                ('ascan_dim',        'Q'),
                ('a_per_bscan',      'Q'),
                ('a_inactive',       'Q'),
                ('b_per_vol',        'Q'),
                ('disp_amin',        'Q'),
                ('disp_amax',        'Q'),
                ('disp_bmin',        'Q'),
                ('disp_bmax',        'Q'),
                ('disp_b_per_vol',   'Q'),
                ('fast_scan_length', 'd'),
                ('slow_scan_length', 'd'),
                ('scan_depth',       'd'),
            ]),
        ]

    def read_header(self, f, header=None):
        header = header or {}

        for parser in self._parsers:
            parser.read(f, header)

        # compatibility additions
        header['zdim'] = header['b_per_vol']
        header['ydim'] = header['ascan_dim']
        header['xdim'] = header['a_per_bscan']
        header['xmin'] = header['disp_bmin']
        header['xmax'] = header['disp_bmax']
        header['ymin'] = header['disp_amin']
        header['ymax'] = header['disp_amax']
        header['zmin'] = 1
        header['zmax'] = header['disp_b_per_vol']
        header['inactive'] = header['a_inactive']
        header['xlength'] = header['fast_scan_length']
        header['ylength'] = header['scan_depth']
        header['zlength'] = header['slow_scan_length']

        return header

    def write_header(self, f, header):
        for parser in self._parsers:
            parser.write(f, header)

    def size_volume(self, header):
        return self._size_volume(header)[0]

    def shape_volume(self, header):
        return self._size_volume(header)[1]

    def _size_volume(self, header):
        zdim = header['disp_b_per_vol']
        ydim = header['disp_amax'] - header['disp_amin']
        xdim = header['disp_bmax'] - header['disp_bmin']

        shape = (zdim, ydim, xdim)
        return (_mul(shape), shape)

    def read_volume(self, f, header):
        (size, shape) = self._size_volume(header)
        header['volume'] = numpy.fromfile(f, dtype=numpy.int8, count=size).reshape(shape)
        return header
