# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.resize(1024, 617)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../ui_img/masico.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setIconSize(QtCore.QSize(32, 32))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(20, 350, 191, 101))
        self.textBrowser.setObjectName("textBrowser")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(630, 80, 370, 370))
        self.label_5.setStyleSheet("border-width: 1px;\n"
"border-style: solid;\n"
"border-color: black;")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 170, 180, 50))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(30, 230, 180, 50))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 290, 180, 50))
        self.pushButton.setObjectName("pushButton")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(220, 80, 370, 370))
        self.label_11.setStyleSheet("border-width: 1px;\n"
"border-style: solid;\n"
"border-color: black;\n"
"")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setEnabled(True)
        self.label_10.setGeometry(QtCore.QRect(410, 40, 42, 28))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(30, 110, 180, 50))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(800, 50, 42, 24))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 27))
        self.menubar.setObjectName("menubar")
        self.menumain = QtWidgets.QMenu(self.menubar)
        self.menumain.setTearOffEnabled(False)
        self.menumain.setToolTipsVisible(False)
        self.menumain.setObjectName("menumain")
        MainWindow.setMenuBar(self.menubar)
        self.actionHELP = QtWidgets.QAction(MainWindow)
        self.actionHELP.setObjectName("actionHELP")
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setCheckable(False)
        self.action.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.action.setFont(font)
        self.action.setObjectName("action")
        self.menumain.addAction(self.action)
        self.menumain.addSeparator()
        self.menubar.addAction(self.menumain.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "检测系统"))
        self.pushButton_5.setText(_translate("MainWindow", "输入图片"))
        self.pushButton_6.setText(_translate("MainWindow", "开始检测"))
        self.pushButton.setText(_translate("MainWindow", "结束"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">输入</span></p></body></html>"))
        self.pushButton_7.setText(_translate("MainWindow", "载入模型"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">输出</span></p></body></html>"))
        self.menumain.setTitle(_translate("MainWindow", "文件(F)"))
        self.actionHELP.setText(_translate("MainWindow", "HELP."))
        self.action.setText(_translate("MainWindow", "退出"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
