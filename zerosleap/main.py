from PySide2.QtWidgets import QApplication

from zerosleap.gui.app import MainWindow

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()