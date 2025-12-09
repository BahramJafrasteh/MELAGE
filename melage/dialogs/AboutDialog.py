__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
import os
from melage.config import *
class about_dialog(QtWidgets.QDialog):
    """
    This calss has been implemented to show image info
    """

    closeSig = pyqtSignal()
    def __init__(self, parent=None, source_folder=''):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("About")
        self.source_folder = source_folder
        self.setupUi()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(about_dialog, self).closeEvent(a0)

    def setupUi(self):
        Dialog = self.window()
        Dialog.setObjectName("Image info")
        Dialog.resize(803, 340)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(10,10,10,10)
        self.grid_main.setObjectName("gridLayout")



        self.video = QVideoWidget(self)
        self.video.resize(500, 500)
        self.video.move(0, 0)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video.sizePolicy().hasHeightForWidth())
        self.video.setMinimumSize(QtCore.QSize(100, 100))
        self.video.setSizePolicy(sizePolicy)
        self.player = QMediaPlayer(self)

        self.player.setVideoOutput(self.video)
        playlist = QMediaPlaylist(self)
        file = os.path.join(self.source_folder, "authors.mp4")
        playlist.addMedia(QMediaContent(QUrl.fromLocalFile(file)))
        playlist.setPlaybackMode(QMediaPlaylist.Loop)


        self.player.setPlaylist(playlist)
        self.grid_main.addWidget(self.video, 1, 0, 1, 4)



        self.splitter = QtWidgets.QSplitter(self)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        self.label = QtWidgets.QLabel(self.splitter)
        self.label.setText('<a href="https://melage.uca.es/about-us/">MELAGE WebSite</a>')
        self.label.setOpenExternalLinks(True)
        self.label2 = QtWidgets.QLabel(self.splitter)
        self.label2.setText('<a href="https://inibica.es/">INiBICA</a>')
        self.label2.setOpenExternalLinks(True)

        self.label3 = QtWidgets.QLabel(self.splitter)
        self.label3.setText('<a href = "mailto: mealge@inibica.es">Support1</a>')
        self.label3.setOpenExternalLinks(True)

        self.label4 = QtWidgets.QLabel(self.splitter)
        self.label4.setText('<a href = "mailto: melage@gamil.com">Support2</a>')
        self.label4.setOpenExternalLinks(True)

        self.logo = QtWidgets.QLabel(self)
        self.logo.setPixmap(QtGui.QPixmap(os.path.join(self.source_folder,"about_logo.png")))

        self.grid_main.addWidget(self.logo, 2, 2, 2, 2, alignment=QtCore.Qt.AlignCenter)


        #self.grid_main.addWidget(self.label, 1, 0, 1, 1)
        self.grid_main.addWidget(self.label, 2, 0, 1, 1)
        self.grid_main.addWidget(self.label2, 3, 0, 1, 1)
        self.grid_main.addWidget(self.label3, 2, 1, 1, 1)
        self.grid_main.addWidget(self.label4, 3, 1, 1, 1)

        self.label5 = QtWidgets.QLabel(self.splitter)
        self.label5.setText('{} ({})'.format(VERSION, VERSION_DATE))
        self.label5.setOpenExternalLinks(True)

        self.copyright = QtWidgets.QLabel(self)
        self.copyright.setText('<a href="https://www.safecreative.org/work/2211222681375-melage">License</a> @COPYRIGHT 2023 ')
        self.copyright.setOpenExternalLinks(True)

        self.grid_main.addWidget(self.label5, 4, 0, 1, 1)
        self.grid_main.addWidget(self.copyright, 4, 1, 1, 1)


    def show(self) -> None:
        self.player.setPosition(0)  # to start at the beginning of the video every time

        #self.video.show()
        self.player.play()
        super(about_dialog, self).show()







def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = about_dialog()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()