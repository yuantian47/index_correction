import os

from imageio import imwrite
import numpy

from PyBROCT import broct
from PyBROCT.render import Renderer
from PyBROCT.matrix_util import lookat, translate, roty, scale, perspective

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description='OCT flythrough generator', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', help='path to OCT file')

args = parser.parse_args()

renderer = Renderer()

renderer.set_threshold(0.53)
renderer.set_brightness(1.5)
renderer.set_transfer_function([0, 0.1, 1])

modelview = lookat(numpy.array([0, 3, 4])*0.9, [0, 0, 0], [0, -1, 0])
modelview = translate([0, -1.2, 0]).dot(modelview)
modelview = modelview.dot(roty(-45))
modelview = modelview.dot(scale([1, 3, 1]))

projection = perspective(30, 1, 0.01, 100)

output_root = os.path.splitext(args.path)[0]
try:
    os.makedirs(output_root)
except FileExistsError:
    pass

volumes = broct.volumes(open(args.path, 'rb'))
(_, data) = next(volumes)
shape = data['volume'].shape
print(shape, [data[k + 'length'] for k in ['x', 'y', 'z']])

fps = 30

laser_rate = 100e3
bscan_rate = laser_rate / (data['xmax'] + data['inactive'])


volume = numpy.zeros(shape) # next(broct.volumes(open(args.path, 'rb'), skip=-1))[1]['volume']

volume_number = 0
frame_number = 0
last_mip_bscan_index = 0
while True:
    bscan_number = int(frame_number / fps * bscan_rate)

    while True:
        bscan_index = bscan_number - volume_number * shape[0]

        mip_bscan_index = min(bscan_index, shape[0] - 1)
        volume[last_mip_bscan_index:mip_bscan_index+1] = data['volume'][last_mip_bscan_index:mip_bscan_index+1]

        last_mip_bscan_index = mip_bscan_index

        if bscan_index < shape[0]:
            break

        last_mip_bscan_index = 0
        volume_number +=1
        try:
            (_, data) = next(volumes)
        except StopIteration:
            raise SystemExit

    print('{}/{}/{}'.format(frame_number, volume_number, bscan_number), end=' ', flush=True)

    renderer.set_volume(volume, True)
    image = renderer.render(modelview, projection, (512, 512))
    image = (image[:, :, 0] * (image[:, :, 3] / 255.0)).astype(numpy.uint8)
    image = numpy.stack([image, image, image, numpy.ones_like(image)], axis=-1)
    image[..., 3] = 255 * (image[..., 0] > 0)

    out = os.path.join(output_root, 'vol{:06d}.png'.format(frame_number))
    imwrite(out, image)

    frame_number += 1
