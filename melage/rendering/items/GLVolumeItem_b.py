from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem

from PyQt5 import QtGui

from .. import shaders

import numpy as np

from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram

__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""

__all__ = ['GLVolumeItem']

class GLVolumeItem(GLGraphicsItem):
    """
    """
    
    
    def __init__(self, sliceDensity=1, smooth=True, glOptions='translucent',
                 intensity_seg=1.0):
        """
        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 4D numpy array (x, y, z, RGBA) with dtype=ubyte.
        sliceDensity    Density of slices to render through the volume. A value of 1 means one slice per voxel.
        smooth          (bool) If True, the volume slices are rendered with linear interpolation 
        ==============  =======================================================================================
        """
        
        self.sliceDensity = 1
        self.smooth = smooth
        self.data = None
        self._needUpload = False
        self.texture = None
        self._first = True
        self.program = []
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.intensitySeg=1


    def setData(self, data):
        data = data[::-1]
        data = np.flip(data, axis=2)
        data = np.transpose(data, [2, 1, 0, 3])

        self.data = data

        self._needUpload = True
        self.update()




    def upload_data(self):

        glEnable(GL_TEXTURE_3D)
        glShadeModel(GL_SMOOTH)

        #glDepthFunc(GL_LESS)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.0, 0.0, 0.0, 1)
        #glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST) #Enable depth testing for z-culling
        glDepthFunc(GL_LEQUAL) #Set the type of depth-test
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)# Nice perspective corrections

        if self.texture is None:
            self.texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture)


        if self.smooth:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        #glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
        glTexParameteri(GL_TEXTURE_3D, GL_GENERATE_MIPMAP, GL_FALSE)# automatic mipmap
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)


        glPixelTransferf(GL_RED_SCALE, 1)
        glPixelTransferf(GL_GREEN_SCALE, 1)
        glPixelTransferf(GL_BLUE_SCALE, 1)
        #glPixelTransferf(GL_ALPHA_SCALE, 1)


        shape = self.data.shape

        ## Test texture dimensions first
        glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        data = np.ascontiguousarray(self.data.transpose((2, 1, 0, 3)))
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception("OpenGL failed to create 3D texture (%dx%dx%d); too large for this hardware." % shape[:3])
        if self._first:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            self._first = True
        else:
            glTexSubImage3D(GL_TEXTURE_3D, 0,0,0,0, shape[0], shape[1], shape[2], GL_RGBA, GL_UNSIGNED_BYTE, data)



        self.lists = {}
        for ax in [0,1,2]:
            for d in [-1, 1]:
                l = glGenLists(1)
                self.lists[(ax,d)] = l
                glNewList(l, GL_COMPILE)
                self.drawVolume(ax, d)
                glEndList()

        self._needUpload = False


                
    def paint(self):

        if self.data is None:
            return


        if self._needUpload:
            self.upload_data()




        glEnable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glColor4f(1, 1, 1, self.intensitySeg)
        glEnable(GL_DEPTH_TEST)
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_FRONT)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_3D)

        glBindTexture(GL_TEXTURE_3D, self.texture)


        view = self.view()
        center = QtGui.QVector3D(*[x/2. for x in self.data.shape[:3]])
        cam = self.mapFromParent(view.cameraPosition()) - center
        #print "center", center, "cam", view.cameraPosition(), self.mapFromParent(view.cameraPosition()), "diff", cam
        cam = np.array([cam.x(), cam.y(), cam.z()])
        self._ax = np.argmax(abs(cam))
        self._d = 1 if cam[self._ax] > 0 else -1
        glCallList(self.lists[(self._ax,self._d)])  ## draw axes


        glDisable(GL_TEXTURE_3D)

        glClear(GL_DEPTH_BUFFER_BIT)
        if len(self.program)>0:
            glUseProgram(0)
                


    def drawVolume(self, ax, d):
        N = 5
        
        imax = [0,1,2]
        imax.remove(ax)
        
        tp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        vp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        nudge = [0.5/x for x in self.data.shape]
        tp[0][imax[0]] = 0+nudge[imax[0]]
        tp[0][imax[1]] = 0+nudge[imax[1]]
        tp[1][imax[0]] = 1-nudge[imax[0]]
        tp[1][imax[1]] = 0+nudge[imax[1]]
        tp[2][imax[0]] = 1-nudge[imax[0]]
        tp[2][imax[1]] = 1-nudge[imax[1]]
        tp[3][imax[0]] = 0+nudge[imax[0]]
        tp[3][imax[1]] = 1-nudge[imax[1]]
        
        vp[0][imax[0]] = 0
        vp[0][imax[1]] = 0
        vp[1][imax[0]] = self.data.shape[imax[0]]
        vp[1][imax[1]] = 0
        vp[2][imax[0]] = self.data.shape[imax[0]]
        vp[2][imax[1]] = self.data.shape[imax[1]]
        vp[3][imax[0]] = 0
        vp[3][imax[1]] = self.data.shape[imax[1]]
        slices = self.data.shape[ax] * self.sliceDensity
        r = list(range(slices))
        if d == -1:
            r = r[::-1]

        glBegin(GL_QUADS)
        tzVals = np.linspace(nudge[ax], 1.0-nudge[ax], slices)
        vzVals = np.linspace(0, self.data.shape[ax], slices)
        for i in r:

            z = tzVals[i]
            w = vzVals[i]
            
            tp[0][ax] = z
            tp[1][ax] = z
            tp[2][ax] = z
            tp[3][ax] = z
            
            vp[0][ax] = w
            vp[1][ax] = w
            vp[2][ax] = w
            vp[3][ax] = w
            
            
            glTexCoord3f(*tp[0])
            glVertex3f(*vp[0])
            glTexCoord3f(*tp[1])
            glVertex3f(*vp[1])
            glTexCoord3f(*tp[2])
            glVertex3f(*vp[2])
            glTexCoord3f(*tp[3])
            glVertex3f(*vp[3])
        glEnd()

