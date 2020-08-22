import os

from imageio import imwrite
import numpy

from PyBROCT import broct

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description='OCT flythrough generator', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', help='path to OCT file')

args = parser.parse_args()

output_root = os.path.splitext(args.path)[0]
try:
    os.makedirs(output_root)
except FileExistsError:
    pass

volumes = broct.volumes(open(args.path, 'rb'))
(_, volume) = next(volumes)
shape = volume['volume'].shape
print(shape, [volume[k + 'length'] for k in ['x', 'y', 'z']])

fps = 30

laser_rate = 100e3
bscan_rate = laser_rate / (volume['xmax'] + volume['inactive'])

mip = numpy.zeros((shape[0], shape[2]))

volume_number = 0
frame_number = 0
last_mip_bscan_index = 0
while True:
    bscan_number = int(frame_number / fps * bscan_rate)

    while True:
        bscan_index = bscan_number - volume_number * shape[0]

        mip_bscan_index = min(bscan_index, shape[0] - 1)
        mip[last_mip_bscan_index:mip_bscan_index+1, :] = numpy.amax(volume['volume'][last_mip_bscan_index:mip_bscan_index+1, :, :], axis=1)

        last_mip_bscan_index = mip_bscan_index

        if bscan_index < shape[0]:
            break

        last_mip_bscan_index = 0
        volume_number +=1
        try:
            (_, volume) = next(volumes)
        except StopIteration:
            raise SystemExit

    bscan = volume['volume'][bscan_index, :, :]

    print('{}/{}/{}'.format(frame_number, volume_number, bscan_number), end=' ', flush=True)

    black = 65
    white = 90
    out = os.path.join(output_root, 'bscan{:06d}.png'.format(frame_number))
    dpi = (1e3 * volume['ylength'] / 2 / bscan.shape[0], 1e3 * volume['zlength'] / bscan.shape[1])
    imwrite(out, (255*numpy.clip((bscan - black) / (white - black), 0, 1)).astype(numpy.uint8), dpi=dpi)

    black = 75
    white = 95
    out = os.path.join(output_root, 'mip{:06d}.png'.format(frame_number))
    dpi = (1e3 * volume['xlength'] / mip.shape[0], 1e3 * volume['zlength'] / mip.shape[1])
    imwrite(out, (255*numpy.clip((mip - black) / (white - black), 0, 1)).astype(numpy.uint8), dpi=dpi)

    frame_number += 1
