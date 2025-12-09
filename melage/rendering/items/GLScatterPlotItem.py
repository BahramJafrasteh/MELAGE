from OpenGL.GL import *
from OpenGL.arrays import vbo
from .. GLGraphicsItem import GLGraphicsItem
from .. import shaders

from PyQt5 import QtGui


import numpy as np
__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""

__all__ = ['GLScatterPlotItem']

class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    def __init__(self, **kwds):
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'translucent')
        self.setGLOptions(glopts)
        self.pos = []
        self.size = 10
        self.intensitySeg = 1
        self.color = [1.0,1.0,1.0,1.0]
        self.pxMode = True
        #self.vbo = {}      ## VBO does not appear to improve performance very much.
        self.setData(**kwds)
        self.shader = None
    
    def setData(self, **kwds):
        """
        """
        args = ['pos', 'color', 'size', 'pxMode']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
            
        args.remove('pxMode')
        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])
                #self.vbo.pop(arg, None)
                
        self.pxMode = kwds.get('pxMode', self.pxMode)
        self.update()

    def initializeGL(self):
        if self.shader is not None:
            return
        
        ## Generate texture for rendering points
        w = 1
        def fn(x,y):
            r = ((x-(w-1)/2.)**2 + (y-(w-1)/2.)**2) ** 0.5
            return 255 * (w/2. - np.clip(r, w/2.-1.0, w/2.))
        pData = np.empty((w, w, 4))
        pData[:] = 255
        pData[:,:,3] = np.fromfunction(fn, pData.shape[:2])
        #print pData.shape, pData.min(), pData.max()
        pData = pData.astype(np.ubyte)
        
        if getattr(self, "pointTexture", None) is None:
            self.pointTexture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pData.shape[0], pData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, pData)
        
        self.shader = shaders.getShaderProgram('pointSprite')

    def paint(self):

        self.setupGLState()

        glEnable(GL_POINT_SPRITE)

        glActiveTexture(GL_TEXTURE0)
        glEnable( GL_TEXTURE_2D )
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
    
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)    ## use texture color exactly
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE )  ## texture modulates current color
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glPixelTransferf(GL_DEPTH_BIAS, 0.5)
        with self.shader:
            #glUniform1i(self.shader.uniform('texture'), 0)  ## inform the shader which texture to use
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnable(GL_DEPTH_TEST)
            try:
                pos = self.pos
                #if pos.ndim > 2:
                    #pos = pos.reshape((-1, pos.shape[-1]))
                glVertexPointerf(pos)
            
                if isinstance(self.color, np.ndarray):
                    glEnableClientState(GL_COLOR_ARRAY)
                    self.color[...,-1]=self.intensitySeg
                    glColorPointerf(self.color)
                else:
                    if isinstance(self.color, QtGui.QColor):
                        glColor4f(*fn.glColor(self.color))
                    else:
                        glColor4f(*self.color)
                
                if not self.pxMode or isinstance(self.size, np.ndarray):
                    glEnableClientState(GL_NORMAL_ARRAY)
                    norm = np.empty(pos.shape)
                    if self.pxMode:
                        norm[...,0] = self.size
                    else:
                        gpos = self.mapToView(pos.transpose()).transpose()
                        pxSize = self.view().pixelSize(gpos)
                        norm[...,0] = self.size / pxSize
                    
                    glNormalPointerf(norm)
                else:
                    glNormal3f(self.size, 0, 0)  ## vertex shader uses norm.x to determine point size
                    #glPointSize(self.size)
                glDrawArrays(GL_POINTS, 0, int(pos.size / pos.shape[-1]))
            finally:
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                #posVBO.unbind()
                ##fixes #145
                glDisable( GL_TEXTURE_2D )
