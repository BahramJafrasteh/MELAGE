import numpy as np

from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
import numpy as np

from OpenGL.GL import *

from PyQt5 import QtGui

__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""

__all__ = ['GLGridItem']

class GLGridItem(GLGraphicsItem):
    """
    """
    
    def __init__(self, size=None, color=(255, 255, 255, 76.5), antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if size is None:
            size = QtGui.QVector3D(20,20,1)
        self.setSize(size=size)
        self.setSpacing(1, 1, 1)
        self.setColor(color)
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()
        
    def size(self):
        return self.__size[:]

    def setSpacing(self, x=None, y=None, z=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
        if spacing is not None:
            x = spacing.x()
            y = spacing.y()
            z = spacing.z()
        self.__spacing = [x,y,z]
        self.update() 
        
    def spacing(self):
        return self.__spacing[:]
        
    def setColor(self, color):
        """By Bahram Jafrasteh"""
        self.color = color #fn.Color(color)
        self.update()

    def setGrid(self, xvals, yvals, zvals):
        """
        By Bahram Jafrasteh
        :param xvals:
        :param yvals:
        :param zvals:
        :return:
        """
        self.xvals = xvals
        self.yvals = yvals
        self.zvals = zvals


    def xyGrid(self):
        """
        By Bahram jafrasteh
        :return:
        """
        for x in self.xvals:
            glColor4f(self.color[0], self.color[1], self.color[2],
                      self.color[3])
            glVertex3f(x, self.yvals[0], 0)
            glVertex3f(x, self.yvals[-1], 0)
            glColor4f(self.color[0], self.color[1], self.color[2],
                      self.color[3])
        for y in self.yvals:
            glVertex3f(self.xvals[0], y, 0)
            glVertex3f(self.xvals[-1], y, 0)
        #############Additional
        """
        for x in self.xvals:
            glColor4f(self.color[0], self.color[1], self.color[2],
                      self.color[3])
            glVertex3f(x, self.yvals[0], self.zvals[-1])
            glVertex3f(x, self.yvals[-1], self.zvals[-1])
            glColor4f(self.color[0], self.color[1], self.color[2],
                      self.color[3])
        for y in self.yvals:
            glVertex3f(self.xvals[0], y, self.zvals[-1])
            glVertex3f(self.xvals[-1], y, self.zvals[-1])

        """

    def xzGrid(self):
        """
        By Bahram jafrasteh
        :return:
        """
        for x in self.xvals:
            glVertex3f(x, 0, self.zvals[0])
            glVertex3f(x, 0, self.zvals[-1])
        for z in self.zvals:
            glVertex3f(self.xvals[0], 0, z)
            glVertex3f(self.xvals[-1], 0, z)

        # Additional
        """
        for x in self.xvals:
            glVertex3f(x, self.yvals[-1], self.zvals[0])
            glVertex3f(x, self.yvals[-1], self.zvals[-1])
        for z in self.zvals:
            glVertex3f(self.xvals[0], self.yvals[-1], z)
            glVertex3f(self.xvals[-1], self.yvals[-1], z)
        """

    def yzGrid(self):
        """
        By Bahram jafrasteh
        :return:
        """
        for y in self.yvals:
            glVertex3f(0, y, self.zvals[-1])
            glVertex3f(0, y, self.zvals[0])

        for z in self.zvals:
            glVertex3f(0, self.yvals[0], z)
            glVertex3f(0, self.yvals[-1], z)


        ## Additional
        """
        
        for y in self.yvals:
            glVertex3f(self.xvals[-1], y, self.zvals[-1])
            glVertex3f(self.xvals[-1], y, self.zvals[0])

        for z in self.zvals:
            glVertex3f(self.xvals[-1], self.yvals[0], z)
            glVertex3f(self.xvals[-1], self.yvals[-1], z)
        """

    def color(self):
        return self.__color

    def paint(self):
        self.setupGLState()
        
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glColor4f(self.color[0], self.color[1], self.color[2],
                  self.color[3])
        glBegin( GL_LINES )
        
        self.xyGrid()
        self.xzGrid()
        self.yzGrid()
        
        glEnd()
