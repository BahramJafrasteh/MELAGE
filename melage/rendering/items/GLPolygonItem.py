from OpenGL.GL import *
from ..GLGraphicsItem import GLGraphicsItem
import numpy as np
from PyQt5 import QtGui


__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""

__all__ = ['GLPolygonItem']


class GLPolygonItem(GLGraphicsItem):
    """

    """

    def __init__(self, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self.colors = [[1,0,1],[0,1,1], [1, 0, 1],[1,0,0], [0,1,0], [0,0,1], [1,1,1],[1,1,0],
        ]


    def update_points(self, polys):
        self.polys = polys

    def paint(self):

        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable( GL_BLEND )
        # glEnable( GL_ALPHA_TEST )
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glDisable(GL_DEPTH_TEST)
        glLineWidth(5)



        #glLineStipple(1, 0xF00F)
        glEnable(GL_LINE_STIPPLE)
        glLineWidth(5.0)

        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)



        r = 0
        for key in self.polys.keys():
            poly = self.polys[key]
            if len(poly)<5:
                continue
            r = r + 1
            if r > 7:
                r = 0
            color = self.colors[r]
            glBegin(GL_LINE_STRIP)
            glColor4f(color[0], color[1], color[2], 1)  # z is green
            for (x, y, z) in poly:
                glVertex3f(x, y, z)
            glEnd()





