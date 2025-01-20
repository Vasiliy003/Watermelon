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

class mywindow(QtWidgets.QMainWindow):
    signal = pyqtSignal(dict)
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.importButton.clicked.connect(self.file_open)
        self.ui.startButton.clicked.connect(self.calculations)
        self.ui.results.setWidgetResizable(True)
        self.ui.results.setFixedHeight(681)

        self.signal.connect(self.create_cards)

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

    def calculations(self):
        thread1 = threading.Thread(target=self.predict_start)
        thread1.start()

    def predict_start(self):
        self.ui.startButton.setEnabled(False)

        name = self.ui.image.pixmap()
        name.save('../temp/source.jpg')

        predictor = Predictor()
        path = '../temp/source.jpg'

        results = predictor.start(path)

        self.ui.results.setEnabled(True)

        self.signal.emit(results)

    def create_cards(self, results):
        print(results)
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
                                        "background-color: #dedede;\n"
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

app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())