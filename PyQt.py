import sys
from PyQt5 import QtCore, QtGui, QtWidgets, QStyleFactory
from secondgui import Ui_Dialog
import numpy as np


class App(Ui_Dialog):
    
    def __init__(self, dialog):
        Ui_Dialog.__init__(self)
        self.setupUi(dialog)
         
        # Connect "add" button with a custom function (FetchInputText)
        self.add.clicked.connect(self.readData)
        
 
    def FetchInputText(self):
        txt = self.lineEdit.text()
        self.listWidget.addItem(txt)
        return txt
            
    def readData(self):
        # 40 MHz kom hier in
        txt = self.lineEdit.text()
        print(txt)
        arraydata =np.load(txt)
        data = arraydata.T
        print(data)
        return data

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QDialog()
 
    prog = App(dialog)
    app.setStyle(QStyleFactory.create('cleanlooks'))
    dialog.show()
    sys.exit(app.exec_())