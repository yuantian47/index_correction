from math import pi, floor, ceil, radians, degrees, tan
from itertools import product, combinations
from time import time

import numpy
from imageio import imwrite

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush, QFontMetricsF, QCursor
from PyQt5.QtCore import Qt, QObject, QPoint, QPointF, QPoint, QLineF, QRectF, QRect, pyqtSignal, pyqtSlot, QThread

from PyBROCT.render.math import perspective2, translate, rotz, roty, rotx, scale, viewport, rotate, transform, lookat
from PyBROCT.render.transformations import decompose_matrix
from PyBROCT.viewer.background import RenderJob
from PyBROCT.viewer.util import quantized_floor, quantized_ceil

import logging
logger = logging.getLogger(__name__)

class VolumeWidget(QWidget):
    def __init__(self, *args, **kwargs):
        self._state = kwargs.pop('state')

        super().__init__(*args, **kwargs)

        self._view = lookat([0, 7, 0], [0, 0, 0], [0, 0, 1])
        self._model = scale([1]*3)
        self._projection = self._viewport = None

        self._interactive = True
        self._annotations = 0
        self._transparency = False
        self._scale_render = 1
        if self._state.display_volume.renderer.type == 'CPU':
            self._scale_display = 4
        else:
            self._scale_display = 1

        self._screen_bounds_padding = 10
        self._screen_bounds_display = None
        self._screen_bounds_render = None
        self._crop_corners = None
        self._full_corners = list(product([-1, 1], repeat=3))

        self._resolution_display = [0, 0]
        self._resolution_render = [0, 0]
        self._render_times = []

        self._drag_button = None
        self._view_history = []
        self._last_pos = None
        self._shift_down = False
        self._ctrl_down = False

        self._state.display_volume.updated.connect(self.update)

        self._state.display_volume.manager.begin_render.connect(lambda j, s='Rendering': self._change_status(s))
        self._state.display_volume.manager.end_render.connect(lambda j, s='Idle': self._change_status(s))
        self._state.display_volume.manager.begin_sync.connect(lambda j, s='Fitlering': self._change_status(s))
        self._state.display_volume.manager.end_sync.connect(lambda j, s='Idle': self._change_status(s))

    def _change_status(self, status):
        self._status_text = status
        self.update()

    def mousePressEvent(self, e):
        self._last_pos = e.pos()
        if self._interactive and not self._drag_button:
            self._drag_button = e.button()
            self._view_history.append(self._view.copy())
            self._shift_down = (QApplication.keyboardModifiers() & Qt.ShiftModifier) == Qt.ShiftModifier

    def mouseReleaseEvent(self, e):
        self._last_pos = e.pos()
        if e.button() == self._drag_button:
            self._drag_button = None

    def mouseMoveEvent(self, e):
        if self._drag_button:
            dx = self._last_pos.x() - e.x()
            if dx > self.width() / 2:
                dx = -self.width() + dx
            elif dx < -self.width() / 2:
                dx = self.width() + dx
            dy = self._last_pos.y() - e.y()
            if dy > self.height() / 2:
                dy = -self.height() + dy
            elif dy < -self.height() / 2:
                dy = self.height() + dy

            if (QApplication.keyboardModifiers() & Qt.ControlModifier) == Qt.ControlModifier:
                f = 25
                dx = quantized_floor(abs(dx), f) * sign(dx)
                dy = quantized_floor(abs(dy), f) * sign(dy)

                self._last_pos = QPoint(self._last_pos.x() - dx, self._last_pos.y() - dy)
            else:
                self._last_pos = e.pos()

        if self._drag_button == Qt.RightButton:
            if self._shift_down:
                # spin
                self._view[:3, :3] = rotz(dy / 100 * pi / 4)[:3, :3] * self._view[:3, :3]
            else:
                # zoom in/out
                dy = max(-10, dy)
                dy = min(10, dy)
                self._view *= scale([(100 - dy) / 100]*3)

            self.update()

        elif (self._drag_button == Qt.LeftButton and self._shift_down) or self._drag_button == Qt.MiddleButton:
            # pan
            self._view = translate([-dx / 200, dy / 200, 0]) * self._view
            self.update()

        elif self._drag_button == Qt.LeftButton and not self._shift_down:
            # tilt
            self._view[:3, :3] = roty(-dx / 50 * pi / 4)[:3, :3] * rotx(-dy / 50 * pi / 4)[:3, :3] * self._view[:3, :3]
            self.update()

        wrap = [None, None]
        if e.x() + 1 >= self.width():
            wrap[0] = e.x() - self.width() + 2
        elif e.x() <= 0:
            wrap[0] = e.x() + self.width() - 2

        if e.y() + 1 >= self.height():
            wrap[1] = e.y() - self.height() + 2
        elif e.y() <= 0:
            wrap[1] = e.y() + self.height() - 2

        if wrap[0] is not None or wrap[1] is not None:
            if wrap[0] is None:
                wrap[0] = e.x()
            if wrap[1] is None:
                wrap[1] = e.y()

            self._last_pos = QPoint(*wrap)

            QCursor.setPos(self.mapToGlobal(self._last_pos))

    def wheelEvent(self, e):
        if self._interactive and not e.angleDelta().isNull():
            dw = -e.angleDelta().y()

            dw = max(-10, dw)
            dw = min(10, dw)
            self._view *= scale([(100 + dw) / 100]*3)
            self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()
        elif self._interactive and e.key() == Qt.Key_Home:
            self._view_history.append(self._view.copy())
            self._view = lookat([0, 7, 0], [0, 0, 0], [0, 0, 1])
            self.update()
        elif e.key() == Qt.Key_A:
            self._annotations = (self._annotations + (-1 if QApplication.keyboardModifiers() & Qt.ShiftModifier == Qt.ShiftModifier else 1)) % 3
            self.update()
        elif e.key() == Qt.Key_T:
            self._transparency = not self._transparency
            self.update()
        elif self._interactive and e.key() == Qt.Key_PageUp: # and self._state.volume_position > 0:
            self._state.read(self._state.ascan_position - 400)
            self.update()
        elif self._interactive and e.key() == Qt.Key_PageDown: # and self._state.volume_position + 1 < self._state.volume_count:
            self._state.seek(self._state.ascan_position + 400)
            self.update()
        elif e.key() == Qt.Key_S:
            (_, render_uv) = self._bounds2uv(self._screen_bounds_render, self._scale_render, clip_screen=False)
            # image = self._state.render_volume.render(self._view * self._model * self._state.aspect_xform, self._projection, self._resolution_render, render_uv)
            # imwrite('test.png', image)
            raise RuntimeError('save not reimplemented yet')
        elif e.key() == Qt.Key_Backspace:
            if self._interactive and self._view_history:
                self._view = self._view_history.pop()
                self.update()
        elif e.key() == Qt.Key_F:
            if self.windowState() == Qt.WindowFullScreen:
                self.setWindowState(Qt.WindowNoState)
            else:
                self.setWindowState(Qt.WindowFullScreen)

    def paintEvent(self, e):
        # ref: https://forums.khronos.org/showthread.php/62124
        # self._projection = perspective(30 * self.height() / 512, self.width() / self.height(), 0.01, 100)
        s = 1/tan(radians(self._state.projection_fov)/2)
        self._projection = perspective2(s * 512 / self.height(), s * 512 / self.width(), 0.01, 100)
        self._viewport = viewport(0, self.height(), self.width(), -self.height())

        self._crop_corners = list(product(*self._state.volume_crop))
        pts = [self._project(c) for c in self._crop_corners]
        top_left = [min([c[i] - self._screen_bounds_padding for c in pts]) for i in range(2)]
        bottom_right = [max([c[i] + self._screen_bounds_padding  for c in pts]) for i in range(2)]

        self._screen_bounds_render = [
            [quantized_floor(x, self._scale_render) for x in top_left],
            [quantized_ceil(x, self._scale_render) for x in bottom_right]
        ]
        self._screen_bounds_display = [
            [quantized_floor(x, self._scale_display) for x in top_left],
            [quantized_ceil(x, self._scale_display) for x in bottom_right]
        ]

        self._resolution_render = ((numpy.array(self._screen_bounds_render[1]) - self._screen_bounds_render[0]) // self._scale_render).astype(numpy.intp)

        painter = QPainter(self)

        # background
        if self._transparency:
            painter.fillRect(self.rect(), Qt.magenta)
        else:
            painter.fillRect(self.rect(), Qt.black)

        # volume
        self._draw_volume(painter)

        # annotations
        if self._annotations > 0:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            self._draw_volume_axes(painter)
            self._draw_volume_bounding_box(painter)

            metrics = QFontMetricsF(painter.font())

            pen = QPen()
            pen.setColor(QColor(255, 255, 255))
            painter.setPen(pen)

            dt = numpy.mean(self._render_times or [0])
            self._render_times = self._render_times[-10:]

            lines = [
                (2, f'{self._state.display_volume.renderer.type}  {1e3*dt:6.1f} ms  {1/(dt+1e-6):6.1f} fps'),
                (2, f'{self._resolution_display[0]} \u00d7 {self._resolution_display[1]} px'.ljust(15) \
                    + f'{self._state.display_volume.shape[0]} \u00d7 {self._state.display_volume.shape[1]} \u00d7 {self._state.display_volume.shape[2]} vx'),
                (1, f'{self._state.volume_position+1} of {self._state.volume_count}'),
                (1, f'{self._resolution_render[0]} \u00d7 {self._resolution_render[1]} px'),
                (2, f'{self._state.acquired_shape_mm[0]:.2f} \u00d7 {self._state.acquired_shape_mm[1]:.2f} \u00d7 {self._state.acquired_shape_mm[2]:.2f} mm'),
                (2, f'{self._state.acquired_shape_vx[0]} \u00d7 {self._state.acquired_shape_vx[1]} \u00d7 {self._state.acquired_shape_vx[2]} vx'),
                (1, self._state.filename),
            ]

            i = 0
            for (lvl, line) in reversed(lines):
                if self._annotations >= lvl:
                    painter.drawText(3, self.height() - i * metrics.height() - metrics.descent() - 2, line)
                    i += 1

            scale_factor, _, angles, translation, _ = decompose_matrix(self._view * self._model)

            lines = [
                (1, f'{degrees(angles[0]):+6.1f}\u00b0  {degrees(angles[1]):+6.1f}\u00b0  {degrees(angles[2]):+6.1f}\u00b0'),
            ]

            i = 0
            for (lvl, line) in reversed(lines):
                if self._annotations >= lvl:
                    painter.drawText(self.width() - 3 - metrics.boundingRect(line).width(), self.height() - i * metrics.height() - metrics.descent() - 2, line)
                    i += 1

        if self._annotations >= 2:
            if self._status_text:
                rect = metrics.boundingRect(self._status_text)
                painter.drawText(self.width() - rect.width() - 3, rect.height(), self._status_text)

    def _project(self, p):
        if isinstance(p, tuple):
            p = list(p)
        elif hasattr(p, 'flat'):
            p = list(p.flat)

        q = transform(self._projection * self._view * self._model * self._state.aspect_xform, p + [1])
        return transform(self._viewport, q / q[3])[:2]

    def _project_qt(self, p):
        return QPointF(*self._project(p))

    def _bounds2uv(self, bounds, scale_factor, clip_screen=True):
        bounds = numpy.array(bounds)
        if clip_screen:
            bounds[:, 0] = numpy.clip(bounds[:, 0], 0, quantized_ceil(self.width(), scale_factor))
            bounds[:, 1] = numpy.clip(bounds[:, 1], 0, quantized_ceil(self.height(), scale_factor))

        uv = ((2 * bounds / [self.width(), self.height()] - 1)).T.tolist()
        uv[1] = [-uv[1][0], -uv[1][1]]

        return (bounds, uv)

    def _draw_volume(self, painter, within_bounds=True, clip_screen=True):
        if within_bounds:
            (bounds, render_uv) = self._bounds2uv(self._screen_bounds_display, self._scale_display, clip_screen=clip_screen)
            draw_rect = QRect(QPoint(*bounds[0]), QPoint(*bounds[1]))
        else:
            render_uv = [(-1, 1), (1, -1)]
            draw_rect = self.rect()

        self._resolution_display = [draw_rect.width() // self._scale_display, draw_rect.height() // self._scale_display]

        job = RenderJob(
            modelview=self._view * self._model * self._state.aspect_xform,
            projection=self._projection,
            resolution=self._resolution_display,
            uv=render_uv,
            rect=draw_rect
        )
        self._state.display_volume.render(job)

        render = self._state.display_volume.last_render
        if render is not None and render.image is not None:
            self._render_times.append(render.duration)
            painter.drawImage(render.rect, QImage(render.image, render.image.shape[1], render.image.shape[0], QImage.Format_RGBA8888))

    def _draw_volume_axes(self, painter):
        painter.save()

        metrics = QFontMetricsF(painter.font())

        scale_factor = decompose_matrix(self._model)[0]

        pen = QPen()
        origin = self._project([0]*3)
        for i in range(3):
            pt = [0]*3
            pt[i] = 1

            # axis line
            color = [0]*3
            color[i] = 255

            pen.setColor(QColor(*color))
            painter.setPen(pen)
            brush = QBrush(QColor(*color), Qt.SolidPattern)
            painter.setBrush(brush)

            pt = self._project(pt)
            painter.drawLine(QPointF(*origin), QPointF(*pt))
            circle = QRectF(-1.5, -1.5, 3, 3)
            circle.translate(*pt)
            painter.drawEllipse(circle)

            # axis label
            pen.setColor(QColor(255, 255, 255))
            painter.setPen(pen)

            text = f'{self._state.cropped_shape_mm[::-1][i]:.2f} mm\n{self._state.cropped_shape_vx[::-1][i]} vx'
            if abs(scale_factor[::-1][i] - 1) > 1e-2:
                text += f'\n({scale_factor[::-1][i]:.2f}x)'

            bounds = metrics.boundingRect(QRectF(-100, -100, 200, 200), Qt.AlignCenter, text)

            eps = 1e-6
            norm = (pt - origin) / (numpy.linalg.norm(pt - origin) + eps)
            shift = numpy.abs([bounds.width(), bounds.height()] / (norm + eps) / 2)
            bounds.translate(*(pt + (min(shift) + 5) * norm))

            painter.drawText(bounds, Qt.AlignCenter, text)

        painter.restore()

    def _draw_volume_bounding_box(self, painter):
        painter.save()

        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        painter.setPen(pen)

        segments = [s for s in combinations(self._crop_corners, 2) if sum([a != b for (a, b) in zip(*s)]) == 1]
        painter.drawLines([QLineF(self._project_qt(a), self._project_qt(b)) for (a, b) in segments])

        if any([abs(x) != 1 for x in sum(self._state.volume_crop, [])]):
            pen.setDashPattern([2, 2])
            painter.setPen(pen)

            segments = [s for s in combinations(self._full_corners, 2) if sum([a != b for (a, b) in zip(*s)]) == 1]
            painter.drawLines([QLineF(self._project_qt(a), self._project_qt(b)) for (a, b) in segments])

        if self._annotations >= 1:
            painter.setRenderHint(QPainter.Antialiasing, False)

            pen.setColor(QColor(255, 255, 0))
            pen.setDashPattern([5, 5])
            painter.setPen(pen)

            painter.drawRect(QRect(QPoint(*self._screen_bounds_render[0]), QPoint(*self._screen_bounds_render[1])))

            if self._annotations >= 2:
                pen.setDashPattern([2, 2])
                painter.setPen(pen)
                painter.drawRect(QRect(QPoint(*self._screen_bounds_display[0]), QPoint(*self._screen_bounds_display[1])))

        painter.restore()
