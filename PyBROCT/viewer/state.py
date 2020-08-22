from math import ceil, floor

import numpy

from PyQt5.QtCore import pyqtSignal, QObject

from PyBROCT.io.reader import ScanReader
from PyBROCT.render.math import scale, translate
from PyBROCT.viewer.background import RenderJob, SyncJob, RenderManager
from PyBROCT.viewer.util import quantized_remainder

import logging
logger = logging.getLogger(__name__)

class VolumeCache:
    def __init__(self, reader, limit):
        self._cache = {}
        self._order = []

        self._reader = reader
        self._limit = limit

    def get(self, idx):
        if idx in self._cache:
            logger.debug(f'loading volume {idx} from cache')
        else:
            logger.debug(f'reading volume {idx} from disk')

            # load from reader
            self._reader.seek_abs(idx)
            self._cache[idx] = self._reader.read()['volume']
            self._order.append(idx)

            # prune cache
            while len(self._cache) > self._limit:
                del self._cache[self._order.pop(0)]

        # lookup in cache
        return self._cache[idx]

    def clear(self):
        self._cache.clear()
        self._order.clear()

class VolumeState(QObject):
    updated = pyqtSignal()

    def __init__(self, master, renderer, name):
        super().__init__()

        self.name = name

        self.max_shape = None

        self._volume = None
        self.renderer = renderer

        self.manager = RenderManager(self.renderer)
        self.manager.end_render.connect(self._on_end_render)

        self.last_render = None

        self.out_of_sync = True

    def _on_end_render(self, job):
        self.last_render = job
        self.updated.emit()

    def render(self, job):
        if not self.out_of_sync and self.last_render is not None and self.last_render.is_same(job):
            return

        if self.out_of_sync:
            self.out_of_sync = False
            self.manager.sync(SyncJob(volume=self.volume))

        self.manager.render(job)

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, v):
        self.out_of_sync = True
        self._volume = v

    @property
    def shape(self):
        return self.volume.shape

class MasterState:
    def __init__(self, filename, reader=None, renderer_cls=None):
        self.filename = filename
        self.reader = reader
        if not self.reader:
            self.reader = ScanReader()
            self.reader.open(filename)

        self._volume_cache = VolumeCache(self.reader, 3)

        self.intervolume_ascans = 0
        self.ascan_rate = 100_000

        self._ascan_position = None

        self.volume_crop = [[-1, 1] for i in range(3)]
        self.aspect_xform = None

        self.cross_sections = []

        self.projection_fov = 30
        self._threshold = 0.5

        self.median_filter = True
        self.persist = True
        self.loop = False

        if not renderer_cls:
            from PyBROCT.render import Renderer
            renderer_cls = Renderer

        self.display_volume = VolumeState(self, renderer_cls(), 'display')
        self.render_volume = VolumeState(self, renderer_cls(), 'render')

        self._composite_volume = numpy.empty(self.reader.shape, numpy.int8)

        if renderer_cls.type == 'CPU':
            self.display_volume.max_shape = [128]*3
            self.render_volume.max_shape = [128]*3
        else:
            self.display_volume.max_shape = [256]*3

        for vs in self._volume_states:
            vs.renderer.set_threshold(self.threshold)
            vs.renderer.set_brightness(1)
            vs.renderer.set_range(0.1, 100)
            vs.renderer.set_transfer_function([0, 0.1, 1])

            if vs.max_shape:
                logger.warning(f'limiting {vs.name} volume to {vs.max_shape} voxels for performance')

        self._update_crop()

    @property
    def _volume_states(self):
        return [self.display_volume, self.render_volume]

    def decompose(self, idx, wall=False):
        if wall:
            a_per_b = self.header['a_per_bscan'] + self.header['a_inactive']
            a_per_v = a_per_b * self.header['b_per_vol'] + self.intervolume_ascans
        else:
            a_per_b = self.header['a_per_bscan']
            a_per_v = a_per_b * self.header['b_per_vol']

        (volumes, idx) = quantized_remainder(idx, a_per_v)
        (bscans, ascans) = quantized_remainder(idx, a_per_b)

        return (volumes, bscans, ascans)

    def compose(self, volumes=0, bscans=0, ascans=0, wall=False):
        if wall:
            a_per_b = self.header['a_per_bscan'] + self.header['a_inactive']
            a_per_v = a_per_b * self.header['b_per_vol'] + self.intervolume_ascans
        else:
            a_per_b = self.header['a_per_bscan']
            a_per_v = a_per_b * self.header['b_per_vol']

        ascans = min([ascans, a_per_b])

        return ascans + bscans * a_per_b + volumes * a_per_v

    def wall2disk(self, idx):
        return self.compose(*self.decompose(idx, wall=True), wall=False)

    def disk2wall(self, idx):
        return self.compose(*self.decompose(idx, wall=False), wall=True)

    def seek(self, position, wall=False):
        logger.info(f'seeking to position {position}')

        if self.ascan_position is None:
            (current_volume_idx, current_bscan_idx, current_ascan_idx) = (None, 0, 0)
        else:
            (current_volume_idx, current_bscan_idx, current_ascan_idx) = self.decompose(self.ascan_position, False)

        (target_volume_idx, target_bscan_idx, target_ascan_idx) = self.decompose(position if not wall else self.wall2disk(position), False)

        logger.debug(f'seeking ({current_volume_idx}, {current_bscan_idx}, {current_ascan_idx}) -> ({target_volume_idx}, {target_bscan_idx}, {target_ascan_idx})')

        # compute prior volume index for unrevealed A-scans
        prior_volume_idx = target_volume_idx - 1
        if prior_volume_idx < 0:
            if self.loop:
                prior_volume_idx += self.volume_count
            else:
                prior_volume_idx = None
        logger.debug(f'prior volume is {prior_volume_idx}')

        dirty_ranges = []

        if current_volume_idx is None or position > self.ascan_position:
            # move forward

            # update the composite volume for prior A-scans
            if target_volume_idx == current_volume_idx:
                # update at current position
                start_bscan = current_bscan_idx
                start_ascan = current_ascan_idx

            elif self.persist and current_volume_idx is not None and prior_volume_idx == current_volume_idx:
                logger.debug(f'finishing previous volume {prior_volume_idx} from B-scan {current_bscan_idx}')

                # finish current volume at B-scan granularity
                current_volume = self._volume_cache.get(current_volume_idx)
                self._composite_volume[current_bscan_idx:, ...] = current_volume[current_bscan_idx:, ...]
                dirty_ranges.append((current_bscan_idx, self._composite_volume.shape[0], True))

                # update at starting position
                start_bscan = 0
                start_ascan = 0

            else:
                # start fresh
                if self.persist and prior_volume_idx is not None:
                    logger.debug(f'displaying previous volume {prior_volume_idx}')
                    self._composite_volume = self._volume_cache.get(prior_volume_idx).copy()
                    dirty_ranges.append((0, self._composite_volume.shape[0], True))
                else:
                    logger.debug('clearing previous volume')
                    self._composite_volume.fill(0)
                    dirty_ranges.append((0, self._composite_volume.shape[0], False))

                # update at starting position
                start_bscan = 0
                start_ascan = 0

            # update the composite volume for current A-scans
            target_volume = self._volume_cache.get(target_volume_idx)
            if start_bscan != target_bscan_idx:
                logger.debug(f'displaying current volume {current_volume_idx} B-scans {start_bscan}-{target_bscan_idx}')
                self._composite_volume[start_bscan:target_bscan_idx, ...] = target_volume[start_bscan:target_bscan_idx, ...]
                dirty_ranges.append((start_bscan, target_bscan_idx, True))

                # update A-scans from start of B-scan
                start_ascan = 0

            if start_ascan != target_ascan_idx:
                logger.debug(f'displaying current volume {current_volume_idx} B-scan {target_bscan_idx} A-scans {start_ascan}-{target_ascan_idx}')
                self._composite_volume[target_bscan_idx, :, start_ascan:target_ascan_idx] = target_volume[target_bscan_idx, :, start_ascan:target_ascan_idx]
                dirty_ranges.append((target_bscan_idx, target_bscan_idx, True))

        elif target_ascan_idx < current_ascan_idx:
            # move backward in time

            if current_volume_idx == target_volume_idx:
                # already loaded
                pass

        for vs in self._volume_states:
            # apply cropping and sampling
            if vs.max_shape is None:
                steps = [1]*3
            else:
                steps = [ceil(n / m) for (n, m) in zip(self.available_shape_vx, vs.max_shape)]
            vs.volume = self._composite_volume[tuple([slice(0, n, s or 1) for (n, s) in zip(self.available_shape_vx, steps)])]

        self._ascan_position = position

    def _update_crop(self):
        planes = []
        for (i, (a, b)) in enumerate(self.volume_crop):
            planes.append([
                [a if j == i else 0 for j in range(3)],
                [1 if j == i else 0 for j in range(3)]
            ])
            planes.append([
                [b if j == i else 0 for j in range(3)],
                [-1 if j == i else 0 for j in range(3)]
            ])

        for vs in self._volume_states:
            vs.renderer.set_cut_planes(planes)

        self.aspect_xform = \
            scale([0.1]*3) * \
            scale(self.available_shape_vx / self.acquired_shape_vx) * \
            scale(self.acquired_shape_mm) * \
            translate([-(a + b) / 2 for (a, b) in self.volume_crop])

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        self._threshold = v
        for vs in self._volume_states:
            vs.renderer.set_threshold(self._threshold)

    @property
    def header(self):
        return self.reader.header

    @property
    def position(self):
        return self.decompose(self.ascan_position, False)

    @property
    def volume_position(self):
        return self.position[0]

    @property
    def volume_count(self):
        return self.reader.count

    @property
    def ascan_count(self):
        return self.volume_count * self.header['b_per_vol'] * self.header['a_per_bscan']

    @property
    def ascan_count_wall(self):
        a_per_b = self.header['a_per_bscan'] + self.header['a_inactive']
        return self.volume_count * (self.header['b_per_vol'] * a_per_b + self.intervolume_ascans)

    @property
    def ascan_position(self):
        return self._ascan_position

    @property
    def ascan_position_wall(self):
        return self.disk2wall(self.ascan_position)

    @property
    def cropped_shape_mm(self):
        return self.cropped_shape_vx / self.acquired_shape_vx * self.acquired_shape_mm

    @property
    def cropped_shape_vx(self):
        return numpy.array([ int(s * (b - a) / 2) for ((a, b), s) in zip(self.volume_crop, self.available_shape_vx)])

    @property
    def available_shape_mm(self):
        return self.available_shape_vx / self.acquired_shape_vx * self.acquired_shape_mm

    @property
    def available_shape_vx(self):
        return numpy.array(self.reader.shape)

    @property
    def acquired_shape_mm(self):
        return numpy.array([self.header[k] for k in ['slow_scan_length', 'scan_depth', 'fast_scan_length']])

    @property
    def acquired_shape_vx(self):
        return numpy.array([self.header[k] for k in ['b_per_vol', 'ascan_dim', 'a_per_bscan']])
