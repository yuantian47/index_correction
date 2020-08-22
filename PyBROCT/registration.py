import numpy
# import scipy.ndimage.filters

def compute(volume):
    # register successive B-scans
    offsets = [[0, 0]]
    last_fft = None
    for i in range(1, volume.shape[0]):
        # xcorr = scipy.ndimage.filters.correlate(volume[i-1], volume[i], mode='wrap')
        if last_fft is None:
            last_fft = numpy.fft.fft2(volume[i-1])
        next_fft = numpy.fft.fft2(volume[i])

        xcorr = numpy.fft.ifft2(numpy.conj(last_fft) * next_fft)
        peak = list(numpy.unravel_index(numpy.argmax(numpy.absolute(xcorr)), xcorr.shape))

        for (j, p) in enumerate(peak):
            if p > volume[i].shape[j] // 2:
                peak[j] -= volume[i].shape[j]

        offsets.append(peak)
        last_fft = next_fft

    # accumulate offsets because each one is between successive B-scans
    offsets = numpy.cumsum(numpy.array(offsets), axis=0)

    return offsets

def apply(volume, offsets, axes=None, inplace=False):
    if not inplace:
        volume = volume.copy()

    for (i, offset) in enumerate(offsets):
        for (j, o) in enumerate(offset):
            if axes is not None and j not in axes:
                continue

            volume[i] = numpy.roll(volume[i], -int(o), axis=j)

    return volume

if __name__ == '__main__':
    import os, sys
    from glob import glob

    import numpy
    # import scipy.ndimage.filters

    from PyBROCT import broct

    input_path = sys.argv[1]
    vidx = int(sys.argv[2])

    axial_factor = 1
    lateral_factor = 5

    print(input_path, end=' ')

    output_path = os.path.join(os.path.dirname(input_path), os.path.splitext(input_path)[0] + f'_{vidx}_reg.broct')
    data = next(broct.volumes(open(input_path, 'rb'), skip=vidx))[1]

    offsets = compute(data['volume'])
    apply(data['volume'], offsets, inplace=True)

    broct.write(open(output_path, 'wb'), data)
    print('->', output_path)
