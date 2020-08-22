from time import time

import numpy

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread

import logging
logger = logging.getLogger(__name__)

class Job:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    @property
    def duration(self):
        return self.end_time - self.start_time

class RenderJob(Job):
    def __init__(self, *args, **kwargs):
        self.modelview = kwargs.pop('modelview')
        self.projection = kwargs.pop('projection')
        self.resolution = kwargs.pop('resolution')
        self.uv = kwargs.pop('uv')

        self.rect = kwargs.pop('rect', None)
        self.image = None

        super().__init__(*args, **kwargs)

    def is_same(self, job):
        return numpy.allclose(self.modelview, job.modelview) and numpy.allclose(self.projection, job.projection) \
            and self.resolution == job.resolution and numpy.allclose(self.uv, job.uv)

class SyncJob(Job):
    def __init__(self, *args, **kwargs):
        if 'volume' in kwargs:
            if 'offset' in kwargs or 'full_size' in kwargs:
                raise RuntimeError('cannot specify "offset" or "full_size" with "volume", use "chunk" instead')

            self.chunk = kwargs.pop('volume')
            self.offset = (0, 0, 0)
            self.full_size = self.chunk.shape

        elif 'chunk' in kwargs:
            self.chunk = kwargs.pop('chunk')
            self.offset = kwargs.pop('offset')
            self.shape = kwargs.pop('size')

        else:
            raise RuntimeError('no volume data, need "volume" or "chunk"')

        self.median_filter = kwargs.pop('median_filter', True)

        super().__init__(*args, **kwargs)

class BackgroundRenderer(QObject):
    render_done = pyqtSignal(RenderJob)
    sync_done = pyqtSignal(SyncJob)

    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer

    @pyqtSlot(RenderJob)
    def render(self, job):
        job.start_time = time()
        job.image = self._renderer.render(job.modelview, job.projection, job.resolution, job.uv)
        job.end_time = time()
        self.render_done.emit(job)

    @pyqtSlot(SyncJob)
    def sync(self, job):
        job.start_time = time()
        self._renderer.set_volume(job.chunk, job.median_filter)
        job.end_time = time()
        self.sync_done.emit(job)

class RenderManager(QObject):
    begin_sync = pyqtSignal(SyncJob)
    end_sync = pyqtSignal(SyncJob)
    begin_render = pyqtSignal(RenderJob)
    end_render = pyqtSignal(RenderJob)

    _launch_render = pyqtSignal(RenderJob)
    _launch_sync = pyqtSignal(SyncJob)

    def __init__(self, renderer, buffered=False, background=True):
        super().__init__()

        self.renderer = renderer
        self.state = None

        if background:
            self._thread = QThread(self)
            self._thread.start()
        else:
            self._thread = None

        # configure background renderer
        self._background_renderer = BackgroundRenderer(self.renderer)
        self._launch_render.connect(self._background_renderer.render)
        self._launch_sync.connect(self._background_renderer.sync)
        self._background_renderer.render_done.connect(self._on_render_done)
        self._background_renderer.sync_done.connect(self._on_sync_done)
        if self._thread:
            self._background_renderer.moveToThread(self._thread)

        self._buffered = buffered
        self._queue = []

        self._idle = True

    def _on_sync_done(self, job):
        self.end_sync.emit(job)
        self._idle = True
        self._work()

    def _on_render_done(self, job):
        self.end_render.emit(job)
        self._idle = True
        self._work()

    def _work(self):
        if not self._queue or not self._idle:
            return

        job = self._queue.pop(0)
        self._idle = False

        # dispatch the next job
        if isinstance(job, SyncJob):
            self.begin_sync.emit(job)
            self._launch_sync.emit(job)
        elif isinstance(job, RenderJob):
            self.begin_render.emit(job)
            self._launch_render.emit(job)
        else:
            raise RuntimeError(f'unknown job type: f{type(job)}')

    def sync(self, job):
        self._queue.append(job)
        self._work()

    def render(self, job):
        if not self._buffered:
            # clear all out prior renders
            self._queue = [j for j in self._queue if not isinstance(j, RenderJob)]

        self._queue.append(job)
        self._work()
