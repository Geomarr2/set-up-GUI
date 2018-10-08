import sys
 
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLineEdit, QInputDialog , QFileDialog, QAbstractButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
 
import random
import time
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 50
        self.top = 50
        self.title = 'RFI Chamber'
        self.width = 640
        self.height = 400
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage('No Status!!!')
        self.initUI()
        
 
    def initUI(self):
        #self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        
        m = PlotCanvas(self, width=5, height=4)
        m.move(0,0)
        # Create textbox

        
        text = 'Button 1'
        hover_text = 'What does button 1 do'
        x=140
        y=100
        x_place =500
        y_place = 0
        self.input_text()
        button_one = self.buttons(text, hover_text, x, y, x_place, y_place)
        self.getText()
        if button_one.clicked():
            self.openFileNamesDialog()
        self.show()
        
    def input_text(self):
        self.textbox = QLineEdit(self)
        self.textbox.move(140, 120)
        self.textbox.resize(280,40)
        
    def getText(self):
        text, okPressed = QInputDialog.getText(self, "Get text","Your name:", QLineEdit.Normal, "")
        if okPressed and text != '':
            print(text)
            
    def openFileNamesDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
            
        
    def buttons(self,text, hover_text, x, y, x_place, y_place):

        # Create a button in the window
        button = QPushButton(text, self)
        button.move(x_place,y_place)
        button.resize(x,y)
        temp = QAbstractButton()
        print(temp)
        button.setToolTip('This s an example button')
        button.clicked.connect(self.on_click)
        return temp
    
    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')
        
class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
 
 
    def plot(self):
        tstart = time.time()
        data = [random.random() for i in range(11)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()
        print ('time:' , (time.time()-tstart))
 
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = App()
    
    sys.exit(app.exec_())