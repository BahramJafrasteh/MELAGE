from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
import numpy as np
from PyQt5 import QtGui


__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""
__all__ = ['GLAxisItem']

class GLAxisItem(GLGraphicsItem):
    """
    """
    
    def __init__(self, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.txt = False
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self.GLViewWidget = None
        self.xO = 0
        self.yO = 0
        self.zO = 0


    def setOriging(self, xO, yO, zO):
        """
        By Bahram jafrasteh
        :param xO:
        :param yO:
        :param zO:
        :return:
        """
        self.xO = xO
        self.yO = yO
        self.zO = zO

    def setSize(self, xmin=None, xmax= None,
                ymin=None, ymax = None,  zmin=None,
                zmax = None, size=None):
        """
        By Bahram Jafrasteh
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            xmin = size.xmin()
            ymin = size.ymin()
            zmin = size.zmin()
            xmax = size.xmax()
            ymax = size.ymax()
            zmax = size.zmax()
        self.__size = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.update()
        
    def size(self):
        return self.__size[:]
    def setGLViewWidget(self, GLViewWidget):
        """
        By Bahram Jafrasteh
        :param GLViewWidget:
        :return:
        """
        self.GLViewWidget = GLViewWidget
    
    def paint(self):

        self.setupGLState()

        glEnable(GL_DEPTH_TEST)
        glLineWidth(5)


        xmin, xmax, ymin, ymax, zmin, zmax = self.size()

        x = np.arange(xmin, xmax + (xmax - xmin) / 10, (xmax - xmin) / 5).astype("int")
        y = np.arange(ymin, ymax + (ymax - ymin) / 10, (ymax - ymin) / 5).astype("int")
        z = np.arange(zmin, zmax + (zmax - zmin) / 10, (zmax - zmin) / 5).astype("int")
        xmax = x[-1];
        ymax = y[-1];
        zmax = z[-1]
        glColor4f(1, 0, 1, 1)  # z is green


        glBegin(GL_LINES)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(self.xO, self.yO, zmax)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(self.xO, self.yO, zmin)
        glEnd()

        glColor4f(1, 0, 0, 1)  # y is red
        glBegin(GL_LINES)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(self.xO, ymax, self.zO)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(self.xO, ymin, self.zO)
        glEnd()

        glColor4f(0, 0, 1, 1)  # x is blue
        glBegin(GL_LINES)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(xmax, self.yO, self.zO)
        glVertex3f(self.xO, self.yO, self.zO)
        glVertex3f(xmin, self.yO, self.zO)
        glEnd()
        if self.txt:
            glColor4f(0, 0, 1, 1)  # x is blue

            strs = [str(i) for i in x[::-1]]
            strs[0] = ''
            for i in range(len(x)):#sagittal
                self.GLViewWidget.renderText(pos=[x[i], self.yO, self.zO - (zmax - zmin) / 10], text=strs[i],
                                             size=[1, 1], rotation=[0, 0, 0])

            glColor4f(0, 0, 0, 1)  # black
            self.GLViewWidget.renderText(
                pos=[self.xO + (xmax - xmin) / 2, self.yO - (ymax - ymin) / 20, self.zO - (zmax - zmin) / 20],
                text='Sagittal',
                size=[1, 1], rotation=[0, 0, 0])

            glColor4f(1, 0, 0, 1)  # y is red
            strs = [str(i) for i in y]
            strs[0]=''
            for i in range(len(y)):#coronal
                self.GLViewWidget.renderText(pos=[self.xO, y[i], self.zO - (zmax - zmin) / 20], text=strs[i],
                                             size=[1, 1], rotation=[0, 0, 0])

            glColor4f(0, 0, 0, 1)  # black
            self.GLViewWidget.renderText(pos=[self.xO - (xmax - xmin) / 30, self.yO + (ymax - ymin) / 2, self.zO],
                                         text='Coronal',
                                         size=[1, 1], rotation=[0, 0, 0])

            glColor4f(1, 0, 1, 1)  # z is green
            strs = [str(i) for i in z[::-1]]
            strs[0] = ''
            for i in range(len(z)):#axial
                self.GLViewWidget.renderText(pos=[self.xO, self.yO - (ymax - ymin) / 20, z[i]], text=strs[i],
                                             size=[1, 1], rotation=[0, 0, 0])

            glColor4f(0, 0, 0, 1)  # black
            self.GLViewWidget.renderText(
                pos=[self.xO - (xmax - xmin) / 30, self.yO - (ymax - ymin) / 30, self.zO + (zmax - zmin) / 2], text='Axial',
                size=[1, 1], rotation=[0, 0, 0])



