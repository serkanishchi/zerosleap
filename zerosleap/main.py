from PySide2.QtWidgets import QApplication

from zerosleap.gui.app import MainWindow


def main():
    """Starts new instance of app."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()