if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(prog='PyBROCT.viewer', description='PyBROCT Viewer', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='path to BROCT file')
    args = parser.parse_args()

    # configure the root logger to accept all records
    import logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(name)s] %(filename)s:%(lineno)d\t%(levelname)s:\t%(message)s')

    # set up colored logging to console
    from rainbow_logging_handler import RainbowLoggingHandler
    console_handler = RainbowLoggingHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName('PyBROCT Viewer')

    from PyBROCT.viewer.state import MasterState
    state = MasterState(args.path)
    state.seek(0)

    from PyBROCT.viewer.viewer import VolumeWidget
    viewer = VolumeWidget(state=state)
    viewer.show()

    app.exec_()
