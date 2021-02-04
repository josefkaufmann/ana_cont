import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication([])
win = QMainWindow()
win.setGeometry(200, 500, 400, 400)
win.setWindowTitle("Test Window")


label = QtWidgets.QLabel(win)
label.setText('Hello World')
label.move(50, 50)
win.show()
label.show()
app.exec_()