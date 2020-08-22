import numpy

from PyBROCT.io.fields import FixedFieldsParser, VariableStringParser, ArrayParser

BROCT_SCAN_RECTANGULAR_VOLUME = 0
BROCT_SCAN_BSCAN = 1
BROCT_SCAN_AIMING = 2
BROCT_SCAN_MSCAN = 3
BROCT_SCAN_RADIAL = 4
BROCT_SCAN_ASCAN = 5
BROCT_SCAN_SPECKLE = 6
BROCT_SCAN_MIXED = 7
BROCT_SCAN_XFAST_YFAST = 8
BROCT_SCAN_XFAST_YFAST_SPECKLE = 9
BROCT_SCAN_SPIRAL = 10

class BroctFormat:
    def __init__(self):
        self._parsers = [
            FixedFieldsParser([
                ('xdim',         'i'),
                ('ydim',         'i'),
                ('zdim',         'i'),
                ('xmin',         'i'),
                ('xmax',         'i'),
                ('ymin',         'i'),
                ('ymax',         'i'),
                ('zmin',         'i'),
                ('zmax',         'i'),
                ('inactive',     'i'),
                ('xlength',      'd'),
                ('ylength',      'd'),
                ('zlength',      'd'),
                ('scan_type',    'i'),
                ('big_xdim',     'i'),
                ('big_xmin',     'i'),
                ('big_xmax',     'i'),
                ('big_inactive', 'i'),
                ('roi',          'i'),
            ]),
            ArrayParser('scan_map', ('zdim',), numpy.int32),
            VariableStringParser('notes', 'i'),
        ]

    def read_header(self, f, header=None):
        header = header or {}

        for parser in self._parsers:
            parser.read(f, header)

        # compatibility additions
        header['depth_samples'] = header['ydim']
        header['ascan_dim'] = header['ydim'] // 2
        header['a_per_bscan'] = header['xdim']
        header['a_inactive'] = header['inactive']
        header['b_per_vol'] = header['zdim']
        header['disp_amin'] = header['ymin']
        header['disp_amax'] = header['ymax']
        header['disp_bmin'] = header['xmin']
        header['disp_bmax'] = header['xmax']
        header['disp_b_per_vol'] = header['zmax'] - header['zmin']
        header['fast_scan_length'] = header['xlength']
        header['slow_scan_length'] = header['zlength']
        header['scan_depth'] = header['ylength']

        return header

    def write_header(self, f, header):
        for parser in self._parsers:
            parser.write(f, header)

    def size_volume(self, header):
        return self._size_volume(header)[0]

    def shape_volume(self, header):
        return self._size_volume(header)[1]

    def _size_volume(self, header):
        xmax = header['xmax']
        xmin = header['xmin']
        ymax = header['ymax']
        ymin = header['ymin']
        zmax = header['zmax']
        zmin = header['zmin']
        # bigXmax = header['big_xmax']
        # bigXmin = header['big_xmin']
        # roi = header['roi']
        scan_type = header['scan_type']

        vSize = (xmax-xmin+1)*(ymax-ymin+1)*(zmax-zmin+1)
        # avgSize = (bigXmax-bigXmin+1)*(ymax-ymin+1)*roi

        # if scan_type == BROCT_SCAN_ASCAN:
            # totalSize = (ymax - ymin +1)*2
        # elif scan_type == 7:
            # vSize = (xmax-xmin+1)*(ymax-ymin+1)*(zdim-roi)
            # totalSize = vSize + avgSize + zdim
        # else:
        if scan_type == BROCT_SCAN_RECTANGULAR_VOLUME:
            totalSize = vSize
        else:
            raise RuntimeError(f'scan type {scan_type} is not supported')

        shape = (zmax-zmin+1, ymax-ymin+1, xmax-xmin+1)
        return (totalSize, shape)

    def read_volume(self, f, header):
        # xmax = header['xmax']
        # xmin = header['xmin']
        # ymax = header['ymax']
        # ymin = header['ymin']
        # zmax = header['zmax']
        # zmin = header['zmin']
        # bigXmax = header['big_xmax']
        # bigXmin = header['big_xmin']
        # roi = header['roi']
        scan_type = header['scan_type']

        # if scan_type == 5:
        #     vSize = (ymax - ymin +1)*2

        #     result['volume'] = numpy.fromfile(f, dtype=numpy.float32, count=vSize).reshape((2, ymax-ymin+1))

        # elif scan_type == BROCT_SCAN_MIXED:
        #     vSize = (xmax-xmin+1)*(ymax-ymin+1)*(zdim-roi)
        #     avgSize = (bigXmax-bigXmin+1)*(ymax-ymin+1)*roi

        #     result['volume'] = numpy.fromfile(f, dtype=numpy.int8, count=vSize).reshape((zdim-roi, ymax-ymin+1, xmax-xmin+1))
        #     result['average'] = numpy.fromfile(f, dtype=numpy.int8, count=avgSize).reshape((roi, ymax-ymin+1, bigXmax-bigXmin+1))
        #     result['map'] = numpy.fromfile(f, dtype=numpy.int32, count=zdim)

        # else:
        if scan_type == BROCT_SCAN_RECTANGULAR_VOLUME:
            (size, shape) = self._size_volume(header)
            header['volume'] = numpy.fromfile(f, dtype=numpy.int8, count=size).reshape(shape)
        else:
            raise RuntimeError(f'scan type {scan_type} is not supported')

        return header
