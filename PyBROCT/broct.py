import sys
import os

def _postprocess(args, img):
    # white and black levels
    img = (img - args.black) / (args.white - args.black)
    img = numpy.clip(img, 0, 1)

    # flips
    if args.flip_vertical:
        img = img[::-1, :]
    if args.flip_vertical:
        img = img[:, ::-1]

    # rotation
    if args.rotate > 0:
        img = numpy.rot90(img, args.rotate % 90)

    return (255 * img).astype(numpy.uint8)

if __name__ == '__main__':
    import os
    from glob import glob
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy
    from imageio import imwrite

    from PyBROCT.io.reader import scans

    parser = ArgumentParser(description='parser for BROCT files', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', nargs='+', help='path to BROCT file')
    parser.add_argument('--volume', '-v', type=int, help='first volume index to read', default=0)
    parser.add_argument('--count', '-c', type=int, help='volumes to read')
    parser.add_argument('--white', '-w', type=int, help='white level', default=100)
    parser.add_argument('--black', '-b', type=int, help='black level', default=54)
    parser.add_argument('--flip-vertical', '-fv', action='store_true', help='flip B-scan vertically')
    parser.add_argument('--flip-horizontal', '-fh', action='store_true', help='flip B-scan horizontal')
    parser.add_argument('--rotate', '-r', type=int, help='rotate B-scan in 90 degree increments', default=0)
    parser.add_argument('--average', '-a', type=int, help='average adjacent B-scans', default=-1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--svp', action='store_true', help='dump an SVP')
    group.add_argument('--scan', nargs='?', type=int, help='B-scan index to dump')
    group.add_argument('--xscan', nargs='?', type=int, help='cross B-scan index to dump')
    group.add_argument('--scan-all', action='store_true', help='dump all B-scans')
    group.add_argument('--mip', action='store_true', help='dump a MIP')

    args = parser.parse_args()

    chunks = max(1, args.average)

    for path in args.path:
        for path in glob(path):
            for (vidx, vol) in scans(path, skip=args.volume, count=args.count):
                data = vol['volume']

                if args.svp:
                    print(path, vidx, 'SVP', end=' ')

                    svp = numpy.mean(data, axis=1)
                    svp = _postprocess(args, svp)

                    out = '{}_{:03d}_svp.png'.format(os.path.splitext(path)[0], vidx)
                    dpi = (1e3 * vol['xlength'] / svp.shape[0], 1e3 * vol['zlength'] / svp.shape[1])
                    imwrite(out, svp, dpi=dpi)
                    print('->', out)
                elif args.mip:
                    print(path, vidx, 'MIP', end=' ')

                    mip = numpy.amax(data, axis=1)
                    mip = _postprocess(args, mip)

                    out = '{}_{:03d}_mip.png'.format(os.path.splitext(path)[0], vidx)
                    dpi = (1e3 * vol['xlength'] / mip.shape[0], 1e3 * vol['zlength'] / mip.shape[1])
                    imwrite(out, mip, dpi=dpi)
                    print('->', out)
                elif args.xscan:
                    xidx = args.xscan
                    print(path, vidx, xidx, end=' ')
                    xscan = data[:, :, xidx].astype(numpy.float32).T

                    xscan = _postprocess(args, xscan)

                    out = '{}_{:03d}_x{:03d}.png'.format(os.path.splitext(path)[0], vidx, xidx)
                    dpi = (1e3 * vol['xlength'] / 2 / xscan.shape[0], 1e3 * vol['zlength'] / xscan.shape[1])
                    imwrite(out, xscan, dpi=dpi)
                    print('->', out)
                else:
                    if args.scan_all:
                        bidxs = range(data.shape[0])
                    elif args.scan is None or args.scan < 0:
                        bidxs = [data.shape[0] // 2]
                    else:
                        bidxs = [args.scan]

                    for bidx in bidxs[::chunks]:
                        print(path, vidx, bidx, end=' ')
                        bscan = data[bidx, :, :].astype(numpy.float32)
                        for i in range(1, chunks):
                            bscan += data[bidx + i, :, :].astype(numpy.float32)
                        bscan /= chunks

                        bscan = _postprocess(args, bscan)

                        out = '{}_{:03d}_{:03d}.png'.format(os.path.splitext(path)[0], vidx, bidx // chunks)
                        dpi = (1e3 * vol['ylength'] / 2 / bscan.shape[0], 1e3 * vol['zlength'] / bscan.shape[1])
                        imwrite(out, bscan, dpi=dpi)
                        print('->', out)
