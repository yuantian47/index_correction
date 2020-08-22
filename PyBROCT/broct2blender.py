import struct

import broct

from scipy.ndimage.filters import median_filter, gaussian_filter


def run(args):
    if args.count == 0:
        return

    for path in args.path:
        for path in glob(path):
            out = '{}.bvox'.format(os.path.splitext(path)[0])
            print(path, '->', out, end=' ')

            bvox = open(out, 'wb')

            # write dummy header
            # ref: http://pythology.blogspot.com/2014/08/you-can-do-cool-stuff-with-manual.html
            bvox.write(struct.pack('=IIII', 0, 0, 0, 0))

            n = 0
            shape = None
            for (idx, vol) in broct.volumes(open(path, 'rb'), skip=args.skip):
                n += 1
                data = vol['volume']
                # data = numpy.swapaxes(data, 0, 2)
                # data = numpy.swapaxes(data, 0, 1)
                # data = numpy.mgrid[0:256,0:256,0:256][2]

                if shape is None:
                    shape = data.shape
                    print(shape)

                data = median_filter(data, size=5)
                for i in range(data.shape[0]):
                    print(i)
                    data[i, ...] = gaussian_filter(data[i, ...], sigma=5)

                data = data.astype(numpy.float32) / 127 / 2 + 0.5
                print(data.min(), data.max(), data.mean())
                # data = (data - args.black) / (args.white - args.black) * 255
                # data /= 255

                data = numpy.clip(data, 0, 1)

                # import cv2
                # cv2.imshow('Bscan', (255*data[::4, ::4, 64]).astype(numpy.uint8))
                # cv2.waitKey()

                print(data.min(), data.max(), data.mean())
                # bvox.write(data.astype('<f4').tostring('F'))
                # for i in range(shape[2]):
                    # bvox.write(data[..., i].astype('<f4').tostring('F'))
                bvox.write(data.astype('<f4').tostring('F'))

                if args.count is not None and args.count > 0:
                    if idx - args.volume + 1 >= args.count:
                        break

            # actually write header
            bvox.seek(0, 0)
            bvox.write(struct.pack('=IIII', shape[0], shape[1], shape[2], n))

            bvox.close()

if __name__ == '__main__':
    import os
    from glob import glob
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy
    import scipy.misc

    parser = ArgumentParser(description='parser for BROCT files', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', nargs='+', help='BROCT files to process')
    parser.add_argument('--skip', '-s', type=int, help='volumes to skip', default=0)
    parser.add_argument('--count', '-c', type=int, help='volumes to read')
    parser.add_argument('--white', '-w', type=int, help='white level', default=100)
    parser.add_argument('--black', '-b', type=int, help='black level', default=40)

    args = parser.parse_args()

    run(args)
