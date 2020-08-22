import logging
logger = logging.getLogger(__name__)

Renderer = None

def _check_render(cls):
    import numpy

    try:
        r = cls()
        r.set_volume(numpy.zeros((2, 2, 2)), True)
        r.render(numpy.eye(4), numpy.eye(4), (1, 1))
    except RuntimeError:
        logger.exception(f'{cls.type} renderer check failed')
        return False
    else:
        logger.debug(f'{cls.type} rendered check passed')
        return True

if not Renderer:
    try:
        logger.debug('loading CUDA renderer')
        from .cuda import CudaRenderer
    except ImportError as e:
        logger.warning(f'CUDA renderer not found: {e}')
    else:
        if _check_render(CudaRenderer):
            Renderer = CudaRenderer

if not Renderer:
    try:
        logger.debug('loading OpenCL renderer')
        from .opencl import OpenClRenderer
    except ImportError as e:
        logger.warning(f'OpenCL renderer not found: {e}')
    else:
        if _check_render(OpenClRenderer):
            Renderer = OpenClRenderer

if not Renderer:
    try:
        logger.debug('loading CPU renderer')
        from .cpu import CpuRenderer as Renderer
    except ImportError as e:
        logger.warning(f'CPU renderer not found: {e}')

if not Renderer:
    raise RuntimeError('no renderer available')
