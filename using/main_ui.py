from fileinput import filename
import os
import threading
from PyQt5.sip import delete
from matplotlib.backends.backend_qt import MainWindow
from PyQt5.QtCore import pyqtSignal

from predictor import Predictor
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from maindesign import Ui_MainWindow
from PIL import Image
import sys

class PredictionThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(dict)

    def __init__(self, predictor, path):
        super().__init__()
        self.predictor = predictor
        self.path = path

    def run(self):
        results = self.predictor.start(self.path, self.progress.emit)
        self.finished.emit(results)

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.importButton.clicked.connect(self.file_open)
        self.ui.startButton.clicked.connect(self.calculations)
        self.ui.results.setWidgetResizable(True)
        self.ui.results.setFixedHeight(681)

        self.ui.progressBar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #363A3D;
                border-radius: 5px;
                text-align: center;
                background-color: #0D0F10;
                color: #363A3D;
            }
            QProgressBar::chunk {
                background-color: #B6F09C;
                border-radius: 5px;
            }
        """)
        self.ui.progressBar.hide()

    def file_open(self):
        name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if name == '':
            return
        image = Image.open(name)
        width, height = image.size
        new_height = 500 * height / width
        new_width = 500
        top = 20
        left = 15
        if new_height > 480:
            new_height = 480
            new_width = 480 * width / height
            left = (530 - new_width) / 2

        self.ui.image.setPixmap(QtGui.QPixmap(name))
        self.ui.image.setGeometry(QtCore.QRect(int(left), int(top), int(new_width), int(new_height)))

        self.ui.startButton.setEnabled(True)

    def update_progress_bar(self, value):
        self.ui.progressBar.setProperty('value', value)

    def calculations(self):
        self.ui.startButton.setEnabled(False)
        self.ui.importButton.setEnabled(False)

        self.ui.progressBar.show()
        self.ui.progressBar.setProperty('value', 0)

        name = self.ui.image.pixmap()
        name.save('../temp/source.jpg')

        self.thread = PredictionThread(Predictor(), '../temp/source.jpg')

        self.thread.progress.connect(self.update_progress_bar)
        self.thread.finished.connect(self.create_cards)

        self.thread.start()

    def create_cards(self, results):
        self.ui.progressBar.setProperty('value', 100)
        for i in reversed(range(self.ui.verticalLayout.count())):
            self.ui.verticalLayout.itemAt(i).widget().deleteLater()
        i = 1
        for path in results:
            result = results[path]
            last = result.rfind('\n')
            emotes = result[:last]
            most_value = result[last+1:]

            self.ui.frame = QtWidgets.QFrame(self.ui.scrollAreaWidgetContents)
            self.ui.frame.setGeometry(QtCore.QRect(10, 10, 441, 311))
            self.ui.frame.setStyleSheet("#frame{\n"
                                        "background-color: #131619;\n"
                                        "border-radius: 20px;\n"
                                        "border: 1px solid #363A3D;\n"
                                        "}")
            self.ui.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.ui.frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.ui.frame.setObjectName("frame")
            self.ui.label = QtWidgets.QLabel(self.ui.frame)
            self.ui.label.setEnabled(True)
            self.ui.label.setGeometry(QtCore.QRect(20, 60, 200, 200))
            self.ui.label.setText("")
            self.ui.label.setPixmap(QtGui.QPixmap(path))
            self.ui.label.setScaledContents(True)
            self.ui.label.setObjectName("label")
            self.ui.label_2 = QtWidgets.QLabel(self.ui.frame)
            self.ui.label_2.setGeometry(QtCore.QRect(20, 0, 421, 51))
            font = QtGui.QFont()
            font.setPointSize(32)
            self.ui.label_2.setFont(font)
            self.ui.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.label_2.setObjectName("label_2")
            self.ui.label_3 = QtWidgets.QLabel(self.ui.frame)
            self.ui.label_3.setGeometry(QtCore.QRect(240, 90, 191, 161))
            font = QtGui.QFont()
            font.setPointSize(10)
            self.ui.label_3.setFont(font)
            self.ui.label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.ui.label_3.setObjectName("label_3")
            self.ui.label_4 = QtWidgets.QLabel(self.ui.frame)
            self.ui.label_4.setGeometry(QtCore.QRect(10, 270, 431, 31))
            font = QtGui.QFont()
            font.setPointSize(16)
            self.ui.label_4.setFont(font)
            self.ui.label_4.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.label_4.setObjectName("label_4")

            _translate = QtCore.QCoreApplication.translate
            self.ui.label_2.setText(_translate("MainWindow", f"Лицо {i}"))
            self.ui.label_3.setText(_translate("MainWindow", emotes))
            self.ui.label_4.setText(_translate("MainWindow", most_value))

            self.ui.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 461, 311*i))
            self.ui.scrollAreaWidgetContents.setMinimumHeight(311*i)
            self.ui.verticalLayout.addWidget(self.ui.frame, 0)
            i+=1
        self.ui.progressBar.hide()
        self.ui.importButton.setEnabled(True)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = mywindow()
    application.show()

    sys.exit(app.exec())