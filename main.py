#!/usr/bin/env python3
"""
main.py
=======
Entry point for the 2P TIFF Simulator GUI.

Usage:
    python main.py
"""

import sys
import os

# Ensure repo root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("2P TIFF Simulator")
    app.setOrganizationName("Person Lab, CU Anschutz")

    # Set a clean monospace base font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
