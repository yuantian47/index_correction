import numpy
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

from PyBROCT.io.reader import scans
from PyBROCT.render.cuda import CudaRenderer
from PyBROCT.render.cpu import CpuRenderer

from PyBROCT.matrix_util import perspective, lookat, rotx, roty, rotz, translate, scale, ortho

path = r"C:\Users\Mark\Downloads\eye.broct"

data = next(scans(path))[1]
vol = data['volume']
# vol = numpy.swapaxes(vol, 0, 2).copy()
print(vol.shape)
vol[:, 1000:, :] = 0
vol[:, :50, :] = 0
# vol = vol[:, ::10, :]
# vol = numpy.flip(vol, axis=1)

# vol[50:150, :, :] = 0
# vol[:, :, 300:500] = 0

# pyplot.imshow(vol[0, :400, :], aspect='auto')
# pyplot.show()
# raise SystemExit

renderers = [cls() for cls in [CudaRenderer, CpuRenderer]]

for renderer in renderers:
    renderer.set_volume(vol, True)
    renderer.set_threshold(0.53)
    renderer.set_brightness(1)

    renderer.set_transfer_function([0, 0.1, 1])

for i in range(180):
    # modelview = numpy.eye(4)
    # modelview[2, 3] = -5
    modelview = lookat(numpy.array([0, 4, 4])*0.7, [0, 0, 0], [0, -1, 0])
    modelview = translate([0, -0.6, 0]).dot(modelview)
    modelview = modelview.dot(roty(i / 100.0 * 180))
    # modelview = modelview.dot(scale([1, 2, 1]))
    # print(modelview)

    # projection = perspective(30, 1, 0.01, 100)
    projection = ortho(-0.3, 0.3, -0.3, 0.3, 0.01, 100)
    # print(projection)

    for renderer in renderers:
        image = renderer.render(modelview, projection, (512, 512))
        # print(image.shape, image.dtype)

        # for i in range(4):
            # cv2.imshow('Render' + str(i), image[:, :, i])

        alpha = image[:, :, 3]
        gray = image[:, :, 0]
        # cv2.imshow(f'{renderer.type} Gray ', gray)

        blend = (gray * (alpha / 255.0)).astype(numpy.uint8)
        print(i)

        # cv2.imshow('Alpha', alpha)
        cv2.imshow(f'{renderer.type} Blend', blend)

    k = cv2.waitKey(-1)
    if k == ord('q'):
        break

