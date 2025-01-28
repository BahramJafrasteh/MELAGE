#!/usr/bin/env python
# -*- coding: utf-8 -*-

__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtWidgets, QtCore, QtGui
from utils.utils import cursorPaint, zonePoint, cursorOpenHand, \
    try_disconnect, cursorErase
import OpenGL.GL as gl
from OpenGL.GL import *
import sys
import numpy as np

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt, QEvent
from PyQt5.QtGui import (QColor, QMatrix4x4, QOpenGLShader, QKeyEvent,
        QOpenGLShaderProgram, QPainter, QWheelEvent)
from PyQt5.QtWidgets import QOpenGLWidget, QWidget, QMenu, QAction
import cv2
import math

from collections import defaultdict
from utils.GMM import GaussianMixture
from sklearn.mixture import GaussianMixture
import pyfftw
from sys import platform
if platform=='darwin':
    from utils.Shaders_120 import vsrc, fsrc, fsrcPaint, vsrcPaint
else:
    from utils.Shaders_330 import vsrc, fsrc, fsrcPaint, vsrcPaint

fft = pyfftw.interfaces.numpy_fft.fft2
ifft = pyfftw.interfaces.numpy_fft.ifft2
fftshift = pyfftw.interfaces.numpy_fft.fftshift
ifftshift = pyfftw.interfaces.numpy_fft.ifftshift

class GLWidget(QOpenGLWidget):
    """
    The main class to visualize image using MELAGE
    """
    ##### the list of signals used in the class #####
    zRotationChanged = pyqtSignal(int) #change rotation angle
    clicked = pyqtSignal() #
    LineChanged = pyqtSignal(object) # line changed
    goto = pyqtSignal(object, object) # if go to signal is activated
    zoomchanged = pyqtSignal(object, object) # zooming is activated
    rulerInfo = pyqtSignal(object, object) # ruler is activated
    sliceNChanged = pyqtSignal(object) # changing slice number
    NewPoints = pyqtSignal(object, object) # adding new points
    mousePress = pyqtSignal(object) # if mouse pressed

    def __init__(self, colorsCombinations, parent=None, currentWidnowName = 'sagittal',
                 imdata=None, type= 'eco',id=0
                 ):
        super(GLWidget, self).__init__(parent)

        self.id = id # unique id of the class
        self.colorsCombinations = colorsCombinations # coloring scheme
        self.imType = type # image type
        self.affine = None # image affine

        self.colorObject = [1,0,0,1] # RGBA
        self.colorInd = 9876 # index of colr
        self.colorInds = [9876] # indices of colors
        self.intensitySeg = 1.0 # intensity of the segmentation
        self.N_rulerPoints = 0 # number of rulers used
        self.showAxis = False # show axis
        self.n_colors = 1 # number of colors used

        self._n_colors_previous = np.inf
        self.smooth = True # smoothing visualization
        self._threshold_image = defaultdict(list) # thresholding image

        self.currentWidnowName = currentWidnowName.lower() # name of the window
        if self.currentWidnowName == 'coronal':
            self.colorv = [1,0,1,1]
            self.colorh = [0,0,1,1]
        elif self.currentWidnowName == 'sagittal':
            self.colorv = [1,0,1,1]
            self.colorh = [1,0,0,1]
        elif self.currentWidnowName == 'axial':
            self.colorh = [0,0,1,1]
            self.colorv = [1,0,0,1]
        self.setContextMenuPolicy(Qt.CustomContextMenu) # enable right click
        if imdata is not None:
            imdata = self.updateImage(imdata)
            #self.setNewImage.emit(imdata.shape)

        self.imSeg = None # segmentaiton image
        self.imSlice = imdata # slice of the image
        self.installEventFilter(self)
        self.program = []
        self.showSeg = True # visualize segmentation or not
        if imdata is not None:
            self.imSeg = np.zeros_like(imdata) #
        else:
            self.imSeg = None


        self.initialState()

    def clear(self):
        """
        clear image slices
        :return:
        """
        self.imSeg = None
        self.imSlice = None

    def _updateScreen(self, screen):
        """
        :param screen:
        :return:
        """
        self._updatePixelRatio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._updatePixelRatio)
            screen.logicalDotsPerInchChanged.connect(self._updatePixelRatio)
    def _updatePixelRatio(self):
        """
        By Bahram Jafrasteh
        :return:
        """
        event = QtGui.QResizeEvent(self.size(), self.size())
        self.resizeEvent(event)

    def resetInit(self):
        # reset class parameters
        self.enabledPan = False # Panning

        self.enabledRotate = False # Rotating
        self.enabledZoom = False # Zooming
        self.enabledGoTo = False # GOTO
        self.enabledRuler = False  # Ruller
        self.points = [] # points selected
        self.rulerPoints = defaultdict(list) # ruler points
        self.guidelines_h = [] # guide lines horizontal
        self.guidelines_v = [] # guide lines vertical
        self.selectedPoints = [] # selected points
        self.colorObject = [1,0,0,1] # color object
        self.colorInd = 9876 # color index
        self.colorInds = [9876] # color indices

    def initialState(self):
        """
        Initial state parameters
        :return:
        """
        self.BandPR1 = 0.0 # bandpass radius 1
        self.BandPR2 = 0.0 # bandpass radius 2
        self.contrast = 1.0 # image contrast
        self.brightness = 0.0 # image brightness
        self.activateSobel = 0 # activate sobel filter
        self.thresholdSobel = 1 # activate sobel threshold
        self.n_colors = 1 # number of colors
        self._n_colors_previous = np.inf # number of colors previous
        self.hamming = False # use hamming filter

        self.senderPoints = None # sender points
        self.senderWindow = None # sender window

        self.setFocusPolicy(Qt.StrongFocus)
        self.clearColor = QColor(Qt.black)
        self.zRot = 0 # rotation in z
        self._threshold_image = defaultdict(list) # image thresholding
        self.enabledPan = False # Panning

        self.enabledRotate = False # Rotating
        self.enabledZoom = False # ZOOM DISABLED
        self.enabledGoTo = False
        self.enabledRuler = False # Ruler
        self.lastPos = QPoint()
        self.points = []
        self.rulerPoints = defaultdict(list)
        self.linePoints = []
        self.guidelines_h = []
        self.guidelines_v = []

        #Initialize camera values
        self.left   = 0
        self.bottom = 0
        self.zoom_level_x = 1
        self.zoom_level_y = 1

        self.ZOOM_IN_FACTOR = 1.05
        self.ZOOM_OUT_FACTOR = 1 / self.ZOOM_IN_FACTOR
        self._kspaces = defaultdict(list)

        self.setCursor(Qt.ArrowCursor)


    def create_histogram(self, img, n_components= 5, plot=False):
        # Colorize image based on Gaussian Mixture Models
        """
        :param img: image
        :param n_components: integer (number of components)
        :param plot:
        :return:
        """
        from sklearn.preprocessing import OneHotEncoder

        if n_components> len(self.colorsCombinations):
            n_components = len(self.colorsCombinations)-1
        #data = img.ravel()
        #min_pixel = 1

        # Fit GMM
        x, y = np.where(img)
        vals = img[tuple(zip(*(np.array((x, y)).T)))]
        data_im = np.array([x, y, vals]).T
        if self.segSlice.max()>0:
            sgm = self.segSlice[tuple(zip(*(np.array((x, y)).T)))].reshape(-1, 1)
            dm = OneHotEncoder(sparse=False).fit_transform(sgm)
            data_im = np.hstack([dm, data_im])
        gmm = GaussianMixture(n_components=n_components, covariance_type = 'diag')
        try:
            pred = gmm.fit_predict(X=(data_im-data_im.mean(0))/data_im.std(0))+1
        except:
            return
        img2 = img.copy()
        img2[tuple(zip(*(np.array((x, y)).T)))] = pred

        list_keys =  [key for key in self.colorsCombinations if self.colorsCombinations[key]!=[]]
        key_comp = np.random.permutation(list_keys)[:n_components]
        # Evaluate GMM
        threshold_image = np.tile(np.expand_dims(img, -1), 3)*0.0

        for com in np.unique(pred):

            color = self.colorsCombinations[key_comp[com-1]]
            ind = img2 == com
            try:
                threshold_image[ind, 0] = color[0]*255.0
                threshold_image[ind, 1] = color[1]*255.0
                threshold_image[ind, 2] = color[2]*255.0
            except:
                print('GMM wrong color {}'.format(com))


        self._threshold_image[self.sliceNum] = threshold_image.astype(np.float64)


    def changeView(self, currentWidnowName, zRot):
        """
        Utilities to help change coronal to sagital vice versa
        :param currentWidnowName: window name
        :param zRot: rotation
        :return:
        """
        self.currentWidnowName = currentWidnowName
        self.initialState()
        self.zRot = zRot
        #self.updateInfo()
        #self.update()

    def updateCurrentImageInfo(self, shape):
        """
        Updating image info using shape information and according to current slice name
        :param shape:
        :return:
        """
        self.imXMax, self.imYMax, self.imZMax = shape

        if self.currentWidnowName == 'sagittal':
            self.labelX = 'C'
            self.colorXID = Qt.red
            self.colorX = [1, 0, 0]

            self.labelY = 'A'
            self.colorYID = Qt.magenta
            self.colorY = [1, 0, 1]

            self.activeDim = 2 # matrix for slicing
            self.imWidth = self.imYMax
            self.imHeight = self.imXMax
            self.imDepth = self.imZMax
        elif self.currentWidnowName == 'axial':
            self.labelX = 'S'
            self.colorXID = Qt.blue
            self.colorX = [0, 0, 1]

            self.labelY = 'C'
            self.colorYID = Qt.red
            self.colorY = [1, 0, 0]

            self.activeDim = 0# matrix for slicing
            self.imWidth = self.imZMax
            self.imHeight = self.imYMax
            self.imDepth = self.imXMax
        elif self.currentWidnowName == 'coronal':
            self.labelX = 'S'
            self.colorXID = Qt.blue
            self.colorX = [0, 0, 1]

            self.labelY = 'A'
            self.colorYID = Qt.magenta
            self.colorY = [1, 0, 1]

            self.activeDim = 1# matrix for slicing
            self.imWidth = self.imZMax
            self.imHeight = self.imXMax
            self.imDepth = self.imYMax

        self.imAr = self.imWidth / self.imHeight # image aspect ratio

        self.maxAllowedDis = math.hypot(self.imWidth/2, self.imHeight/2)/1.0 # maximum allowed distance
        ############################################################################
        self.coord = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.vertex = [(0, 0), (0, self.imHeight), (self.imWidth, self.imHeight), (self.imWidth, 0)]



        #Initialize camera values
        self.left   = 0
        self.bottom = 0
        self.zoom_level_x = 1
        self.zoom_level_y = 1
        self.ZOOM_IN_FACTOR = 1.05
        self.ZOOM_OUT_FACTOR = 1 / self.ZOOM_IN_FACTOR

    def updateInfo(self, imSlice = None, segSlice = None, tract = None, sliceNum = None, shape = None,
                   initialState = False, imSpacing=None):
        """
        Updaing class according to given information
        :param imSlice: image slice
        :param segSlice: segmentation slice
        :param tract: tractography
        :param sliceNum: slice number
        :param shape: shape
        :param initialState: boolean
        :param imSpacing: image spacing
        :return:
        """
        self.sliceNum = sliceNum
        self.imSlice = imSlice
        self.segSlice = segSlice

        self.imSpacing = imSpacing


        #if imdata is not None:
            #self.imdata = self.updateImage(imdata)
        #self.setNewImage.emit(shape)
        if initialState:
            self.initialState()
            self.updateCurrentImageInfo(shape)
            self.UpdatePaintInfo()

        self._kspaces = defaultdict(list)

        self.makeObject()

    def ShowContextMenu(self, pos):
        """
        Context Menu for the current class when panning allowed
        :param pos:
        :return:
        """
        menu = QMenu("Edit")

        escape_action = QAction("Cancel")
        escape_action.triggered.connect(self.escapePolygon)

        menu.addAction(escape_action)
        menu.exec_(self.mapToGlobal(pos))

    def ShowContextMenu_ruler(self, pos):
        """
        Using RULER
        :param pos:
        :return:
        """
        menu = QMenu("Ruler")
        remove_action = QAction("Remove")
        self.key_min_ruler, length, pos_ruler = self._compute_distance_ruler(pos)
        lenght = self.imSpacing[0]*length
        distance_action = QAction("Segment Length {:.2f}mm".format(lenght))
        try:
            angle = self.rulerPoints[self.key_min_ruler]['angle']
        except:
            angle = 0
            pos_ruler = [0, 0]
        angle_action = QAction("Angle {:.2f} degrees".format(angle))
        xy_action = QAction("Center X {:.1f}, Y {:.1f}".format(pos_ruler[0], pos_ruler[1]))
        send_action = QAction("Send to Table")
        remove_action.triggered.connect(self.removeRuler)
        menu.addAction(xy_action)
        menu.addAction(distance_action)
        menu.addAction(angle_action)
        menu.addAction(remove_action)
        menu.addAction(send_action)


        action = menu.exec_(self.mapToGlobal(pos))
        if action == send_action:
            #['ImType', 'Area', 'Perimeter', 'Slice', 'WindowName', 'pos_ruler']
            if length==0:
                return
            vals = []
            vals.append('')
            if self.imType == 'eco':
                vals.append('0')
            else:
                vals.append('1')
            vals.append("{:.2f} mm".format(length))
            vals.append("{:.2f}\N{DEGREE SIGN}".format(angle))
            vals.append(str(self.sliceNum))
            vals.append(self.currentWidnowName)
            vals.append("{:.2f},{:.2f}".format(pos_ruler[0], pos_ruler[1]))
            vals.append('')
            self.rulerInfo.emit(vals, 0)

    def ShowContextMenu_gen(self, pos):
        """
        Context MENU in case of generating contour from lines
        :param pos:
        :return:
        """
        menu = QMenu("ContourGen")
        empty_action = QAction("Empty")
        empty_action.triggered.connect(self.removeRuler)
        gen_action = QAction("GenerateAction")

        menu.addAction(gen_action)
        menu.addAction(empty_action)

        action = menu.exec_(self.mapToGlobal(pos))
        if action == gen_action:
            self.LineChanged.emit([[], [], False, True])
        elif action==empty_action:
            self.linePoints = []
            self.LineChanged.emit([[], self.colorInd, True, False])



    def ShowContextMenu_contour(self, pos):
        """
        Context Menu for contouring including center of contouring, perimeter, interpolation, etc.
        :param pos:
        :return:
        """
        from utils.utils import point_in_contour
        menu = QMenu("Ruler")
        x, y = self.to_real_world(pos.x(), pos.y())
        try:
            color = self.segSlice[int(y), int(x)]
        except:
            color = 0
        try:
            if color in self.colorInds or 9876 in self.colorInds:
                area, perimeter, centerXY, WI_index = point_in_contour(self.segSlice.copy(), (x,y), color)
                area = (self.imSpacing[0]**2)*area
                perimeter = (self.imSpacing[0])*perimeter
            else:
                area, perimeter, centerXY, WI_index = 0, 0, [0, 0], None
        except:
            area = 0.0
            perimeter = 0.0
            centerXY = [0,0]
        area_action = QAction("Surface {:.2f} mm\u00b2".format(area))
        perimeter_action = QAction("Perimeter {:.2f} mm".format(perimeter))
        xy_action = QAction("Center X {:.1f}, Y {:.1f}".format(centerXY[0], centerXY[1]))
        send_action = QAction("Send to Table")
        interploateadd_action = QAction("Add to interploation")


        menu.addAction(xy_action)
        menu.addAction(area_action)
        menu.addAction(perimeter_action)
        menu.addAction(send_action)
        menu.addSeparator()
        menu.addAction(interploateadd_action)
        menu.addSeparator()

        action = menu.exec_(self.mapToGlobal(pos))
        if action == send_action:
            #['ImType', 'Area', 'Perimeter', 'Slice', 'WindowName', 'CenterXY']
            if area==0:
                return
            vals = []
            vals.append('')
            if self.imType == 'eco':
                vals.append('0')
            else:
                vals.append('1')
            vals.append("{:.2f} mm\u00b2".format(area))
            vals.append("{:.2f} mm".format(perimeter))
            vals.append(str(self.sliceNum))
            vals.append(self.currentWidnowName)
            vals.append("{:.2f},{:.2f}".format(centerXY[0], centerXY[1]))
            vals.append('')
            self.rulerInfo.emit(vals, 0)
        return





    def _compute_distance_ruler(self, pos):
        '''
        GET information from RULER
        :param pos:
        :return:
        '''
        key_min = None
        length = 0
        x, y = self.lastPressRuler
        #x, y =  pos
        try:
            min_dist = np.inf
            for key in self.rulerPoints.keys():
                if key is None:
                    continue
                if len(self.rulerPoints[key]['center'])!=3:
                    continue
                a, b, c = self.rulerPoints[key]['center']
                x1, y1, _ = self.rulerPoints[key]['points'][0]
                x2, y2, _ = self.rulerPoints[key]['points'][-1]
                dist_center = (x-(x1+x2)/2.0)**2. + (y-(y1+y2)/2.)**2.
                dist_line = abs(a * y + b * x + c) / np.sqrt(a + b ** 2)
                dist = dist_line + 0.1*np.sqrt(dist_center)
                if dist < min_dist:
                    min_dist = dist
                    key_min = key
                    length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        except Exception as e:
            pass
        return key_min, length, [pos.x(), pos.y()]

    def removeRuler(self):
        """
        Remover one of the lines from ruler
        :return:
        """
        try:
            self.rulerPoints.pop(self.key_min_ruler)
            self.update()
        except Exception as e:
            pass

    def escapePolygon(self):
        """
        When panning menu is activated
        :return:
        """
        self.points = []


    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.update()

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def minimumSizeHint(self):
        return QSize(50, 50)


    def _endIMage(self, x, y):
        """
        check if we are at the end of the image
        :param x: input x direction
        :param y: input y direction
        :return:
        """
        xmax = self.imWidth
        ymax = self.imHeight
        endIm = False
        if x<0:
            endIm = True
            x = 0
        elif x>xmax:
            endIm = True
            x = xmax
        if y <0:
            endIm = True
            y = 0
        elif y > ymax:
            endIm = True
            y = ymax
        return endIm, x, y

    def _endWindow(self, xw, yw):
        """
        check if we are at the end of the the curretn winow
        :param xw: x direction
        :param yw: y direction
        :return:
        """
        xwmax = 0.95*self.width()
        ywmax = 0.95*self.height()
        xwmin = 0.00*self.width()
        ywmin = 0.00*self.height()
        state = 0
        if xw <xwmin or xw>xwmax:
            state = 1
        if yw <ywmin  or yw > ywmax:
            if state != 0:
                state = 3
            else:
                state = 2
        return state


    def pan(self, dx, dy):
        """
        Panning throug image using dx and dy
        :param dx:
        :param dy:
        :return:
        """
        left_p = self.left
        right_p = self.right
        bottom_p = self.bottom
        top_p = self.top
        dx, dy = dx*100/self.width(), dy*100/self.height()
        self.left -= dx #
        self.right -= dx #
        self.bottom -= dy
        self.top -= dy #

        halfPoint = self.to_real_world(self.width()/2, self.height()/2)

        # displacement from the image center
        displacement = math.hypot(halfPoint[0]-self.imWidth/2, halfPoint[1]-self.imHeight/2)

        if displacement > self.maxAllowedDis: # panning should not be done outside of the image
            self.left = left_p
            self.right = right_p
            self.bottom = bottom_p
            self.top = top_p

        self.update()


    def fromRealWorld(self, mouseXWorld, mouseYWorld):
        """
        convert to real world from mouse position
        :param mouseXWorld: mouse x position
        :param mouseYWorld: mouse y position
        :return:
        """
        def convertR(xr, yr): # convert back x and y from rotation
            x_p = (xr-cx)*ca - (yr - cy)*sa+cx
            y_p = (xr-cx)*sa + (yr - cy)*ca+cy
            return x_p, y_p
        # rotation angle
        angle = self.zRot/16
        if angle !=0:
            cx, cy = self.imWidth / 2, self.imHeight / 2
            angle_r = math.radians(angle)
            sa = math.sin(angle_r)
            ca = math.cos(angle_r)
            mouseXWorld, mouseYWorld = convertR(mouseXWorld, mouseYWorld)
        xp = (mouseXWorld - self.left)/self.zoomed_width
        yp = (mouseYWorld - self.bottom) / self.zoomed_height
        return xp*self.width(), yp*self.height()


    def to_real_world(self, x, y):
        """
        convert to real world
        :param x:
        :param y:
        :return:
        """
        def convertR(xr, yr): # convert back x and y from rotation
            x_p = (xr-cx)*ca - (yr - cy)*sa+cx
            y_p = (xr-cx)*sa + (yr - cy)*ca+cy
            return x_p, y_p

        # rotation center
        angle = self.zRot/16

        # scale back the window to image data
        mouseXWorld = (x / self.width()) * self.zoomed_width + self.left
        mouseYWorld = (y/self.height())*self.zoomed_height + self.bottom
        if angle !=0:
            cx, cy = self.imWidth / 2, self.imHeight / 2
            angle_r = math.radians(angle)
            sa = -1*math.sin(angle_r)
            ca = math.cos(angle_r)
            mouseXWorld, mouseYWorld = convertR(mouseXWorld, mouseYWorld)


        return mouseXWorld, mouseYWorld


    def createProgram(self, id, vsrc, fsrc):
        """
        Create shader program using id and other shader information
        :param id:
        :param vsrc:
        :param fsrc:
        :return:
        """
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        vshader = QOpenGLShader(QOpenGLShader.Vertex, self)
        vshader.compileSourceCode(vsrc)
        if not vshader.isCompiled():
            print(vshader.log())

        fshader = QOpenGLShader(QOpenGLShader.Fragment, self)
        fshader.compileSourceCode(fsrc)

        if not fshader.isCompiled():
            print(fshader.log())

        self.program.append(QOpenGLShaderProgram())
        self.program[id].addShader(vshader)
        self.program[id].addShader(fshader)

        #self.program.bindAttributeLocation('in_Vertex',
         #       self.PROGRAM_VERTEX_ATTRIBUTE)
        #self.program.bindAttributeLocation('vertTexCoord',
         #       self.PROGRAM_TEXCOORD_ATTRIBUTE)
        self.program[id].link()
        self.program[id].bind()
        #################### SHADER ID #################
        self.program[id].vertPosAttrId = self.program[id].attributeLocation("in_Vertex")
        self.program[id].vertTexCoordAttrId = self.program[id].attributeLocation("vertTexCoord")

        self.program[id].vertModelViewAttrId = self.program[id].uniformLocation("g_matModelView")
        self.program[id].texSamplerId = self.program[id].uniformLocation("tex")

        self.program[id].texSamplerId2 = self.program[id].uniformLocation("u_input")

        self.program[id].u_transformSizeID = self.program[id].uniformLocation("u_transformSize")
        self.program[id].u_subtransformSizeID = self.program[id].uniformLocation("u_subtransformSize")

        self.program[id].R = self.program[id].uniformLocation("R")
        self.program[id].G = self.program[id].uniformLocation("G")
        self.program[id].B = self.program[id].uniformLocation("B")



        glUniform1i(
            self.program[id].texSamplerId,  # texture sampler uniform ID */
            0)  # value of sampler */

        glUniform1i(
            self.program[id].texSamplerId2,  # texture sampler uniform ID */
            1)  # value of sampler */


        self.program[id].release()
        self.program[id].removeAllShaders()
        return True


    def wheelEvent(self, event: QWheelEvent):
        """
        Whele event
        :param event:
        :return:
        """
        if self.enabledZoom:
            x = event.x()
            y = event.y()
            xr, yr=self.to_real_world(x, y)
            endim = self._endIMage(xr, yr)
            if endim[0]:
                x, y = self.fromRealWorld(endim[1], endim[2])
            deltaAngY = event.angleDelta().y()
            self.scaleF = self.ZOOM_IN_FACTOR if deltaAngY > 0 else self.ZOOM_OUT_FACTOR if deltaAngY < 0 else 1

            if 0.0<self.zoom_level_y*self.scaleF<5 or 0.0<self.zoom_level_x*self.scaleF<5:
                self.updateScale(x, y, self.scaleF, self.scaleF)
        else:
            # scroll images
            deltaAngY = event.angleDelta().y()
            NexSlice = -1 if deltaAngY > 0 else 1 if deltaAngY < 0 else 1
            self.sliceNChanged.emit(self.sliceNum+NexSlice)


    def updateScale(self, x, y, scaleFX, scaleFY):
        """
        update zoom level
        :param x:
        :param y:
        :param scaleFX:
        :param scaleFY:
        :return:
        """
        self.zoom_level_x *= scaleFX
        self.zoom_level_y *= scaleFY

        # locate mouse and to real world
        # mouseXWorld, mouseYWorld, mouseX, mouseY = self.to_real_world(x, y)
        mouseX = x / self.width()  # x location in the window
        mouseY = y / self.height()  # y location in the window
        mouseX = 0.5; mouseY = 0.5
        mouseXWorld = self.left + mouseX * self.zoomed_width
        mouseYWorld = self.bottom + mouseY * self.zoomed_height

        # update zoom width and heigth
        self.zoomed_width *= scaleFX
        self.zoomed_height *= scaleFY

        self.left = mouseXWorld - mouseX * self.zoomed_width
        self.right = mouseXWorld + (1 - mouseX) * self.zoomed_width
        self.bottom = mouseYWorld - mouseY * self.zoomed_height
        self.top = mouseYWorld + (1 - mouseY) * self.zoomed_height
        #if not self.enabledCircle:
        #    self.zoomchanged.emit(self._radius_circle/abs(self.to_real_world( 1, 0)[0] - self.to_real_world(0, 0)[0]), True)
        self.update()

    def initializeGL(self):
        """
        Initialize GL
        :return:
        """
        self.createProgram(0, fsrc=fsrc, vsrc=vsrc)
        self.createProgram(1, fsrc=fsrcPaint, vsrc=vsrcPaint)

        if self.imSlice is None:
            return

        self.UpdatePaintInfo()
        self.makeObject()


    def UpdatePaintInfo(self):
        """
        Update infroamtion of the iamge
        :return:
        """
        self.windowAR = self.width() / (self.height()+0.000001) #windows aspect ratio
        imWidth = self.imWidth
        imHeight = self.imHeight

        ratioX = imWidth / self.width()
        ratioY = imHeight / (self.height()+0.000001)
        if self.width() > self.height():
            if imWidth <= imHeight:

                self.zoomed_height = imHeight
                # self.zoomed_height /= self.imAr
                self.bottom = 0
                self.top = self.bottom + self.zoomed_height

                # rescale to the image width
                self.zoomed_width = ratioY * self.width()

                # image left position
                self.left = imWidth / 2 - (self.zoomed_height * self.windowAR - self.zoomed_width / 2)

                # image right position
                self.right = self.left + self.zoomed_width
            else:
                if ratioX < 1 and ratioY > 1:
                    self.zoomed_height = imHeight
                    self.bottom = 0  #
                    self.top = self.bottom + self.zoomed_height

                    self.zoomed_width = ratioY * self.width()
                    self.left = imWidth / 2 - (self.zoomed_height * self.windowAR - self.zoomed_width / 2)
                    self.right = self.left + self.zoomed_width


                else:
                    self.zoomed_width = imWidth
                    self.left = 0  # -imWidth * (self.windowAR - 1) / 2
                    self.right = self.left + self.zoomed_width

                    self.zoomed_height = ratioX * self.height()
                    self.bottom = imHeight / 2 - (self.zoomed_width / self.windowAR - self.zoomed_height / 2)
                    self.top = self.bottom + self.zoomed_height


        elif self.width() <= self.height():

            if imWidth <= imHeight:
                if ratioX < 1 and ratioY > 1 or ratioX / ratioY < 1:
                    self.zoomed_height = imHeight
                    self.bottom = 0
                    self.top = self.bottom + self.zoomed_height

                    self.zoomed_width = ratioY * self.width()

                    self.left = imWidth / 2 - (self.zoomed_height * self.windowAR - self.zoomed_width / 2)
                    self.right = self.left + self.zoomed_width


                else:
                    self.zoomed_width = imWidth
                    self.left = 0
                    self.right = self.left + self.zoomed_width

                    self.zoomed_height = ratioX * self.height()
                    self.bottom = imHeight / 2 - (self.zoomed_width / self.windowAR - self.zoomed_height / 2)
                    self.top = self.bottom + self.zoomed_height


            else:

                self.zoomed_width = imWidth
                self.left = 0
                self.right = self.left + self.zoomed_width

                self.zoomed_height = ratioX * self.height()
                #self.zoomed_height = 2*imWidth/self.imAr

                self.bottom = imHeight / 2 - (self.zoomed_width / self.windowAR - self.zoomed_height / 2)
                self.top = self.bottom + self.zoomed_height
        #if self.currentWidnowName != 'axial' and self.imType == 't1':
         #   self.updateScale( self.width()//2, self.height()//2, 1, 0.5)
        glEnable(GL_NORMALIZE)  # light normalization

    def drawImage(self):

        # pre requisites and drawImage
        self.drawImagePre()


        # draw additional
        self.drawImPolygon()  # draw imFreeHand



        if len(self.rulerPoints.keys()) >= 1:
            self.DrawRulerLines()

        if len(self.linePoints) > 3:
            self.DrawLines(self.linePoints)


        if len(self.guidelines_h) > 0:
            self.DrawLines(self.guidelines_h, self.colorh)
        if len(self.guidelines_v)>0:
            self.DrawLines(self.guidelines_v, self.colorv)

        # end iamges
        self.drawImageEnd()


    def drawImagePre(self):
        """
        Draw Image pre-requisites
        :return:
        """
        #glPushMatrix()
        #glPushAttrib(
          #  GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT | GL_ENABLE_BIT
           # | GL_DEPTH_BUFFER_BIT)


        ################## static GL data ##################
        # use the program object
        glUseProgram(self.program[0].programId())
        #self.program[0].link()
        #self.program[0].bind()
        # set rotation view translation matrix
        # model-view-projection matrix
        #glUniformMatrix4fv(self.vertModelViewAttrId, 1, GL_FALSE, mvpMatrix)
        # texture
        m = QMatrix4x4()

        m.ortho( self.left, self.right, self.top, self.bottom, -1, 1 )

        m.translate(self.imWidth/2, self.imHeight/2, 0.0)

        m.rotate(self.zRot / 16.0, 0.0, 0.0, 1.0)

        m.translate(-self.imWidth/2, -self.imHeight/2, 0.0)

        self.mvpMatrix = m.copyDataTo()


        glUniformMatrix4fv(self.program[0].vertModelViewAttrId, 1, GL_FALSE, self.mvpMatrix)

        glUniform1fv(self.program[0].u_transformSizeID, 1,self.imWidth)
        glUniform1fv(self.program[0].u_subtransformSizeID, 1, self.imHeight)

        #self.program[0].setUniformValue('g_matModelView', m)

        # thresholding
        threshold_loc = glGetUniformLocation(self.program[0].programId(), "threshold")
        glUniform1f(threshold_loc, 0)

        # contrast
        contrast_loc = glGetUniformLocation(self.program[0].programId(), "contrastMult")
        glUniform1f(contrast_loc, 1)

        # brightness
        brightnessAdd_loc = glGetUniformLocation(self.program[0].programId(), "brightnessAdd")
        glUniform1f(brightnessAdd_loc, self.brightness)

        # soble filter
        activateSobel_loc = glGetUniformLocation(self.program[0].programId(), "sobel")
        glUniform1i(activateSobel_loc, self.activateSobel)
        activateSobel_loc = glGetUniformLocation(self.program[0].programId(), "sobel_threshold")
        glUniform1f(activateSobel_loc, self.thresholdSobel)

        minRad_loc = glGetUniformLocation(self.program[0].programId(), "iResolution")
        glUniform2fv(minRad_loc, 1, [self.width(), self.height()])
        maxRad_loc = glGetUniformLocation(self.program[0].programId(), "maxRadius")
        glUniform1f(maxRad_loc, self.thresholdSobel)

        ilum_loc = glGetUniformLocation(self.program[0].programId(), "Ilum")
        glUniform1f(maxRad_loc, 0.8)

        mouse_pos_loc = glGetUniformLocation(self.program[0].programId(), "mousePos")
        glUniform2fv(mouse_pos_loc, 1, [self.lastPos.x(), self.height()-self.lastPos.y()] )

        # deinterlace
        deinterlace = [self.imHeight, 0, 0]
        deinterlace_loc = glGetUniformLocation(self.program[0].programId(), "deinterlace")
        glUniform3fv(deinterlace_loc, 1, deinterlace)

        self.program[0].enableAttributeArray(self.program[0].vertTexCoordAttrId)
        self.program[0].setAttributeArray(self.program[0].vertTexCoordAttrId, self.coord)
        self.program[0].enableAttributeArray(self.program[0].vertPosAttrId)
        self.program[0].setAttributeArray(self.program[0].vertPosAttrId, self.vertex)

        glEnable(GL_TEXTURE_2D)

        glActiveTexture(GL_TEXTURE0)

        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawArrays(GL_QUADS, 0, 4)
        glDisable(GL_TEXTURE_2D)
        glDisableClientState(GL_VERTEX_ARRAY)
        #glBindTexture(GL_TEXTURE_2D, 0)
        self.program[0].disableAttributeArray(self.program[0].vertTexCoordAttrId)
        self.program[0].disableAttributeArray(self.program[0].vertPosAttrId)


    def drawImageEnd(self):
        glUseProgram(0)  # necessary to release the program


    def paintGL_start(self):

        glClearColor(self.clearColor.redF(), self.clearColor.greenF(),
                self.clearColor.blueF(), self.clearColor.alphaF())


        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)

        glDisable(GL_CULL_FACE) # disable backface culling
        glDisable(GL_LIGHTING) # disable lighting
        glDisable(GL_DEPTH_TEST) # disable depth test

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

    def paintGL(self):
        """
        Painting GL
        :return:
        """
        if self.imSlice is None:
            return
        self.paintGL_start()
        self.drawImage()
        self.paintGL_end()

    def subpaintGL(self,points, windowname):
        """
        reserved function for mutual painting
        :param points:
        :param windowname:
        :return:
        """
        self.senderPoints = None
        self.senderWindow = None
        self.update()

    def paintGL_end(self):
        glClear(GL_DEPTH_BUFFER_BIT)

        #glFlush()
        #swa
        #glCallList(1)
        if self.showAxis:
            self.draw_axis()
        self.drawLabel()



    def draw_axis(self):
        """
        Draw axis
        :return:
        """
        offset = 0.05
        x0, y0, x1, y1 = 0, 0.0, 1,1
        glPushAttrib(GL_CURRENT_BIT)
        glLineStipple(1, 0xF00F)
        glEnable(GL_LINE_STIPPLE)
        glLineWidth(1.0)
        #Set The Color To Blue One Time Only
        glBegin(GL_LINES)
        glColor3f(self.colorX[0], self.colorX[1], self.colorX[2])
        glVertex2f(x0+offset, y0)
        glVertex2f(x1, y0)
        glVertex2f(x0-offset, y0)
        glVertex2f(-x1, y0)
        glColor3f(self.colorY[0], self.colorY[1], self.colorY[2])
        glVertex2f(x0, y0-offset)
        glVertex2f(x0, -y1)
        glVertex2f(x0, y0+offset)
        glVertex2f(x0, y1)
        #glVertex2f(x1, -y1)
        #glVertex2f(x0, y1)


        glEnd()
        glPopAttrib()
        # glEndList()
        glDisable(GL_LINE_STIPPLE)

    def drawLabel(self):
        """

        :return:
        """

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.save()

        font = painter.font()
        font.setPointSize(font.pointSize()*5)
        #font.setPixelSize(font.pixelSize()*1)
        painter.setRenderHint(QPainter.Antialiasing)
        font.setBold(True)

        #pen = QPen(self.colorXID, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        #painter.setPen(pen)

        painter.setPen(self.colorXID)
        painter.drawText(int(self.width()/2.1), int(0.02*self.height()),self.width(),self.height(),Qt.AlignCenter,
                                   self.labelX)
        painter.drawText(-int(self.width()/2.1), int(0.02*self.height()),self.width(),self.height(),Qt.AlignCenter,
                                   self.labelX)


        painter.setPen(self.colorYID)


        painter.drawText(int(0.02*self.width()), -int(self.height()/2.1),self.width(),self.height(),Qt.AlignCenter,
                                   self.labelY)
        painter.drawText(int(0.02*self.width()), int(self.height()/2.1),self.width(),self.height(),Qt.AlignCenter,
                                   self.labelY)

        painter.setPen(Qt.white)
        painter.drawText(-int(self.width()/2.3), -int(self.height()/2.1),self.width(),self.height(),Qt.AlignCenter,
                                   self.currentWidnowName)


        painter.restore()

    def takescreenshot(self, area=None, width=0, height=0):
        """
        This function has been designed to make screenshot from GL
        :param area:
        :param width:
        :param height:
        :return:
        """
        if self.imSlice is None:
            return None
        if area=='whole':
            self.makeObject()
        else:
            self.paintGL()
        if width==0:
            width = self.width()
        if height==0:
            height = self.height()
        val = max(height, width)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        z = glReadPixels(0, 0, val, val, GL_RGBA,
                         GL_FLOAT)[::-1]
        if val == width:
            z = z[val-height:width, 0:width]
        else:
            z = z[0:height, 0:width]

        return z

    def resizeGL(self, width, height):
        if self.imSlice is None:
            return
        side = min(width, height)
        #Set viewport
        glViewport((width- side) // 2, (height-side) // 2, self.imWidth,
                self.imHeight)

    def mousePressEvent(self, event):
        self.mousePress.emit(event)
        self.lastPos = event.pos()



        if self.enabledRotate:
            self.lastPressPos = self.lastPos
            self.pressZone = zonePoint(event.x(), event.y(), self.width() / 2, self.width() / 2)
        elif self.enabledRuler:
            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]

            self.lastPressRuler = [x,y]

            if event.button()==Qt.LeftButton:
                self.rulerPoints[self.N_rulerPoints] = defaultdict(list)
                self.rulerPoints[self.N_rulerPoints]['points'] = []
                self.rulerPoints[self.N_rulerPoints]['perpendicular1'] = []
                self.rulerPoints[self.N_rulerPoints]['perpendicular2'] = []
                self.rulerPoints[self.N_rulerPoints]['center'] = []
                self.rulerPoints[self.N_rulerPoints]['points'].append([x, y, self.sliceNum])

        if self.enabledGoTo:
            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.goto.emit([x, y, self.sliceNum], [self.currentWidnowName, self.imType])


    def updateCursor(self, event, x, y, dx, dy,endIm):
        """
        Update Cursor
        :param event:
        :param x:
        :param y:
        :param dx:
        :param dy:
        :param endIm:
        :return:
        """
        if self.zoom_level_x < 1 and self.zoom_level_y < 1 and not endIm:
            state = self._endWindow(event.x(), event.y())
            if state != 0:
                if state == 1:  # xpanning
                    self.pan(-dx, 0)
                    xWind, yWind = self.fromRealWorld(x, y)
                    xyG = self.mapToGlobal(QPoint(int(xWind), int(yWind)))  # map from current window to global window
                    self.cursor().setPos(xyG.x(), xyG.y())
                elif state == 2:  # y panning
                    self.pan(0, -dy)
                    xWind, yWind = self.fromRealWorld(x, y)
                    xyG = self.mapToGlobal(QPoint(int(xWind), int(yWind)))  # map from current window to global window
                    self.cursor().setPos(xyG.x(), xyG.y())
                elif state == 3:  # x and y panning (corners)
                    self.pan(-dx, -dy)
                    xWind, yWind = self.fromRealWorld(x, y)
                    xyG = self.mapToGlobal(QPoint(int(xWind), int(yWind)))  # map from current window to global window
                    self.cursor().setPos(xyG.x(), xyG.y())

    def mouseMoveEvent(self, event):
        def compute_line(x, m, b):
            return m*x+b
        #print(self.cursor().pos())

        if self.enabledRuler: # if RULER is activated
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x , y  = self.to_real_world(event.pos().x(), event.pos().y())


            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            if 'points' in self.rulerPoints[self.N_rulerPoints]:
                self.rulerPoints[self.N_rulerPoints]['points'].append([x, y, self.sliceNum])
                if len(self.rulerPoints[self.N_rulerPoints]['points'])>=2:
                    self.rulerPoints[self.N_rulerPoints]['points'] = [self.rulerPoints[self.N_rulerPoints]['points'][0],
                                                            self.rulerPoints[self.N_rulerPoints]['points'][-1]]
                    x1, y1 = self.rulerPoints[self.N_rulerPoints]['points'][0][0], self.rulerPoints[self.N_rulerPoints]['points'][0][1]
                    x2, y2 = self.rulerPoints[self.N_rulerPoints]['points'][-1][0], self.rulerPoints[self.N_rulerPoints]['points'][-1][1]
                    self.rulerPoints[self.N_rulerPoints]['center'] = [(x1+x2)/2.0, (y1+y2)/2.0, self.sliceNum]
                    if (y2-y1) != 0:
                        m_prime = -(x2-x1)/(y2-y1)
                        if m_prime != 0:
                            # a, b, c, ax+by+c = 0
                            m = -1.0/m_prime
                            bline= y1 - m*x1
                            self.rulerPoints[self.N_rulerPoints]['center'] = [1, -m,
                                                                              -bline]
                        else:
                            self.rulerPoints[self.N_rulerPoints]['center'] = [0,1,
                                                                              -x1]


                        b = y1-m_prime*x1
                        delta = 5
                        if abs(m_prime)>1:
                            delta /=abs(m_prime)
                        ruler_perpendicular1 = [(x1-delta, compute_line(x1-delta, m_prime, b), self.sliceNum),
                                                     (x1, compute_line(x1, m_prime, b), self.sliceNum),
                                                     (x1+delta, compute_line(x1+delta, m_prime, b), self.sliceNum)]
                        b = y2 - m_prime * x2
                        ruler_perpendicular2 = [(x2-delta, compute_line(x2-delta, m_prime, b), self.sliceNum),
                                                     (x2, compute_line(x2, m_prime, b), self.sliceNum),
                                                     (x2+delta, compute_line(x2+delta, m_prime, b), self.sliceNum)]

                    else:
                        delta = 5
                        ruler_perpendicular1 = [(x1, y1 - delta, self.sliceNum), (x1, y1, self.sliceNum),
                                                    (x1, y1 + delta, self.sliceNum)]
                        ruler_perpendicular2 = [(x2, y2 - delta, self.sliceNum), (x2, y2, self.sliceNum),
                                                    (x2, y2 + delta, self.sliceNum)]
                        self.rulerPoints[self.N_rulerPoints]['center'] = [1, 0,-y1]
                    self.rulerPoints[self.N_rulerPoints]['perpendicular1'] = ruler_perpendicular1
                    self.rulerPoints[self.N_rulerPoints]['perpendicular2'] = ruler_perpendicular2

            self.updateCursor(event, x, y, dx, dy,endIm[0])

            # update the window
            self.update()

        elif event.buttons() & Qt.LeftButton:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            if self.enabledPan:
                self.pan(dx, dy)
            #self.rotateBy(8 * dy, 8 * dx, 0)
        elif event.buttons() & Qt.RightButton:
            dx = abs(event.x() - self.lastPos.x())

            if self.enabledRotate:
                pZone =zonePoint(event.x(), event.y(), self.lastPressPos.x(), self.lastPressPos.y())
                if self.pressZone == 1:
                    if pZone == 2 or pZone == 1:
                        self.setZRotation(self.zRot - 1 * dx)
                    elif pZone == 4 or pZone == 3:
                        self.setZRotation(self.zRot + 1 * dx)
                elif self.pressZone == 2:
                    if pZone == 1:
                        self.setZRotation(self.zRot + 1 * dx)
                    elif pZone == 3:
                        self.setZRotation(self.zRot - 1 * dx)
                elif self.pressZone == 3:
                    if pZone == 2:
                        self.setZRotation(self.zRot + 1 * dx)
                    elif pZone == 4:
                        self.setZRotation(self.zRot - 1 * dx)
                elif self.pressZone == 4:
                    if pZone == 1:
                        self.setZRotation(self.zRot - 1 * dx)
                    elif pZone == 3:
                        self.setZRotation(self.zRot + 1 * dx)

        self.lastPos = event.pos()

        if self.enabledGoTo:
            x, y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.goto.emit([x, y, self.sliceNum], [self.currentWidnowName, self.imType])

    def keyPressEvent(self, event: QKeyEvent):

        if event.key() == Qt.Key_Control:
            self.enabledRotate = True
            self.enabledZoom = True
            try_disconnect(self)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        #key options
        if event.key() == Qt.Key_0:

            self.updateEvents()
            self.setCursor(Qt.ArrowCursor)
            try_disconnect(self)
        elif event.key() == Qt.Key_1:  # ImFreeHand

            self.updateEvents()


            self.setCursor(cursorPaint())
            try_disconnect(self)
            try:
                self.customContextMenuRequested.connect(self.ShowContextMenu)
            except Exception as e:
                pass
        elif event.key() == Qt.Key_2:  # Panning

            self.updateEvents()
            self.enabledPan = True
            self.setCursor(cursorOpenHand())
            try_disconnect(self)
        elif event.key() == Qt.Key_3:  # Erasing
            self.setCursor(cursorOpenHand())
            self.updateEvents()

            self.setCursor(cursorErase())
            try_disconnect(self)
        elif event.key() == Qt.Key_4:  # ImPaint
            self.updateEvents()
            self.setCursor(cursorPaint())
        elif event.key() == Qt.Key_Control:
            try_disconnect(self)
            if self.enabledCircle:
                self.zoomchanged.emit(self._radius_circle/abs(self.to_real_world( 1, 0)[0] - self.to_real_world(0, 0)[0]), True)
            self.enabledZoom = False
            try:
                self.customContextMenuRequested.connect(self.ShowContextMenu)
            except Exception as e:
                pass

    def mouseReleaseEvent(self, event):
        """
        Mouse release events
        :param event:
        :return:
        """
        self.clicked.emit()



        if self.enabledRuler: # if ruler option is activated
            if 'points' in self.rulerPoints[self.N_rulerPoints]:
                if len(self.rulerPoints[self.N_rulerPoints]['points'])>1:
                    #self.rulerPoints.append(self.rulerPoints[0])
                    #compute_distance = compute_distance_between_two_points(self.rulerPoints)
                    endl = self.rulerPoints[self.N_rulerPoints]['points'][1]
                    startl = self.rulerPoints[self.N_rulerPoints]['points'][0]
                    angl = math.atan2(startl[1] - endl[1], startl[0] - endl[0]) * 180 / np.pi
                    if angl<0:
                        angl = abs(angl)
                    else:
                        angl = 180-angl

                    self.rulerPoints[self.N_rulerPoints]['angle'] = angl
                    #print(angl)
                    self.N_rulerPoints += 1


    def drawImPolygon(self): # draw polygons
        def changeAccordingToWN(points, windowname):
            if windowname == 'coronal':
                if self.currentWidnowName == 'sagittal':
                    points = np.array(points)[:,[2,0,1]]
                    return points
                if self.currentWidnowName == 'axial':
                    points = np.array(points)[:, [2, 0, 1]]
                    return points

        glPushAttrib(GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)


        glUseProgram(self.program[1].programId())
        glUniformMatrix4fv(self.program[1].vertModelViewAttrId, 1, GL_FALSE, self.mvpMatrix)



        self.program[1].enableAttributeArray(self.program[1].vertTexCoordAttrId)
        self.program[1].setAttributeArray(self.program[1].vertTexCoordAttrId, self.coord)
        self.program[1].enableAttributeArray(self.program[1].vertPosAttrId)
        self.program[1].setAttributeArray(self.program[1].vertPosAttrId, self.vertex)

        # set brightness/contrast
        color_location = glGetUniformLocation(self.program[1].programId(), "my_color")
        glUniform4fv(color_location, 1, self.colorObject)


        glDisable(GL_LIGHTING)
        glNormal3f(0.5, 1.0, 1.0)
        glLineWidth(1.0)
        glClearColor(1.0, 0.0, 0.0, 1.0)

        # antialias
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glUniform4fv(color_location, 1, self.colorObject)
        if len(self.points)>1:
            self.NewPoints.emit(self.points, self.currentWidnowName)

            glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
            #glColor3f(1.0, 0.0, 0.0)
            for (x, y, z) in self.points:

                glVertex2f(x, y)
            glEnd()

        if len(self.selectedPoints)>0:

            glPointSize(20)
            glBegin( GL_POINTS )  # These vertices form a point
            #glColor3f(1.0, 0.0, 0.0)
            for (x, y, z) in self.selectedPoints:

                glVertex2f(x, y)
            glEnd()


        if self.senderPoints is not None: # arrange points arrray according to the windows information
            points = changeAccordingToWN(self.senderPoints, self.senderWindow)
            glLineWidth(1)
            glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
            #glColor3f(1.0, 0.0, 0.0)
            for (x, y, z) in points:
                glVertex2f(x, y)
            glEnd()
            glLineWidth(1.0)

        glUseProgram(0)
        glPopAttrib()
        # glPopMatrix()
        glMatrixMode(GL_MODELVIEW)



    def DrawLines(self, points, colorv=[1,0,0, 1], width_line= 1):
        """
        Draw lines on the screen
        :param points:
        :param colorv:
        :param width_line:
        :return:
        """
        glPushMatrix()
        glPushAttrib(GL_CURRENT_BIT)

        glUseProgram(self.program[1].programId())
        glUniformMatrix4fv(self.program[1].vertModelViewAttrId, 1, GL_FALSE, self.mvpMatrix)

        self.program[1].enableAttributeArray(self.program[1].vertTexCoordAttrId)
        self.program[1].setAttributeArray(self.program[1].vertTexCoordAttrId, self.coord)
        self.program[1].enableAttributeArray(self.program[1].vertPosAttrId)
        self.program[1].setAttributeArray(self.program[1].vertPosAttrId, self.vertex)



        # color location
        color_location = glGetUniformLocation(self.program[1].programId(), "my_color")
        glUniform4fv(color_location, 1, colorv)
        #glLineStipple(1, 0x00FF)  # [1] dashed lines
        glEnable(GL_LINE_STIPPLE)
        #glDisable(GL_LIGHTING)
        glEnable(GL_LIGHTING)
        glNormal3f(0.5, 1.0, 1.0)
        glLineWidth(width_line)

        glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
        glColor3f(1.0, 1.0, 0.0)  # Red
        for (x, y, z) in points:
            glVertex3f(x, y, z)
        glEnd()

        glUseProgram(0)
        glDisable(GL_LINE_STIPPLE)
        glPopAttrib()
        glPopMatrix()




    def DrawRulerLines(self):
        """
        Draw ruler lines
        :return:
        """
        for ke in self.rulerPoints.keys():
            if not hasattr(self.rulerPoints[ke], 'keys'):
                continue
            for key in self.rulerPoints[ke].keys():
                if key == 'center' or key == 'angle':
                    continue
                points = self.rulerPoints[ke][key]

                glPushMatrix()
                glPushAttrib(GL_CURRENT_BIT)

                glUseProgram(self.program[1].programId())
                glUniformMatrix4fv(self.program[1].vertModelViewAttrId, 1, GL_FALSE, self.mvpMatrix)

                self.program[1].enableAttributeArray(self.program[1].vertTexCoordAttrId)
                self.program[1].setAttributeArray(self.program[1].vertTexCoordAttrId, self.coord)
                self.program[1].enableAttributeArray(self.program[1].vertPosAttrId)
                self.program[1].setAttributeArray(self.program[1].vertPosAttrId, self.vertex)



                # color location
                color_location = glGetUniformLocation(self.program[1].programId(), "my_color")
                glUniform4fv(color_location, 1, [1,0,0, 1])

                glDisable(GL_LIGHTING)
                glNormal3f(0.5, 1.0, 1.0)
                glLineWidth(1.0)

                glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
                glColor3f(1.0, 1.0, 0.0)  # Red
                for (x, y, z) in points:
                    glVertex2f(x, y)
                glEnd()

                glUseProgram(0)

                glPopAttrib()
                glPopMatrix()




    def makeObject(self):
        """
        Create object before painting
        :return:
        """
        activate_kspace = False

        if self.imSlice is None:
            return
        if self.BandPR1 > 0 or self.contrast!=1.0 or self.hamming: # if image enhancement is active
            activate_kspace = True
        imslice = self.imSlice
        if activate_kspace:
            if self.sliceNum in self._kspaces.keys():
                kspace = np.copy(self._kspaces[self.sliceNum])
            else:
                kspace = fftshift(fft(ifftshift(imslice)))
                self._kspaces[self.sliceNum] = np.copy(kspace)


            # band pass filter
            if self.BandPR1 > 0:
                from utils.utils import computeAnisotropyElipse
                self.insideElipse = computeAnisotropyElipse(kspace)
                #SigmaSquared = (self.imWidth*self.BandPR1/10)** 2
                #g = np.array([np.exp(-(np.arange(0,self.imWidth) - self.imWidth/2)** 2 / SigmaSquared)]*self.imHeight)
                #TwoDGauss = g*g
                #HighPass = fftshift(TwoDGauss);
                #kspace*=TwoDGauss

                r = np.hypot(*kspace.shape) / 2 * (self.BandPR1/2)
                r2 = self.BandPR2*r
                r2=0
                rows, cols = np.array(kspace.shape, dtype=int)
                a, b = np.floor(np.array((rows, cols)) / 2).astype(np.int)
                y, x = np.ogrid[-a:rows - a, -b:cols - b]
                #xx = kspace.shape[0] // 2
                #yy = kspace.shape[1] // 2
                #xpx = 1
                # = np.copy(kspace[xx-xpx:xx+xpx, yy-xpx:yy+xpx])


                notMask = self.insideElipse(x, y, r2)
                valcenter = np.copy(kspace[notMask])
                mask = self.insideElipse(x, y, r)

                kspace[mask] = (self.BandPR2-0.5)*2*kspace[mask]
                #kspace[xx - xpx:xx + xpx, yy - xpx:yy + xpx] = valcenter
                kspace[notMask] = valcenter




            if self.contrast!=1.0:

                x = kspace.shape[0] // 2
                y = kspace.shape[1] // 2
                kspace[x-2:x+2, y-2:y+2] *= self.contrast

            if self.hamming:

                x, y = kspace.shape
                window = np.outer(np.hamming(x), np.hamming(y))
                kspace*=window

            vis = False
            if vis:
                kspaceAbs = np.absolute(kspace)
                if kspaceAbs.max() > 0:
                    scaling_c = np.power(10., -3)
                    np.log1p(kspaceAbs * scaling_c, out=kspaceAbs)
                    # normalize between 0 and 255
                    fmin = float(kspaceAbs.min())
                    fmax = float(kspaceAbs.max())
                    if fmax != fmin:
                        coeff = fmax - fmin
                        kspaceAbs[:] = np.floor((kspaceAbs[:] - fmin) / coeff * 255.)


                #imslice = np.require(kspaceAbs, np.uint8)
                imslice = kspaceAbs
            else:
                imslice = np.absolute(fftshift(ifft(ifftshift(kspace))))





        ##############################################################################
        glEnable(GL_TEXTURE_2D) # Enable texturing
        glEnable(GL_COLOR_MATERIAL)

        self.textureID = glGenTextures(1) # Obtain an id for the texture
        glBindTexture(GL_TEXTURE_2D, self.textureID) # Set as the current texture
        if self.smooth:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)# automatic mipmap
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)



        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.imWidth, self.imHeight, 0,
                             #GL_RGB, GL_UNSIGNED_BYTE,
                             #self.imdata)

        glPixelTransferf(GL_RED_SCALE, 1)
        glPixelTransferf(GL_GREEN_SCALE, 1)
        glPixelTransferf(GL_BLUE_SCALE, 1)

        if self.n_colors>1:
            if self.sliceNum not in self._threshold_image or self.n_colors!= self._n_colors_previous\
                    or activate_kspace:
                self.create_histogram(imslice, self.n_colors)
                self._n_colors_previous = self.n_colors
            imslice = self._threshold_image[self.sliceNum].copy()
            imslice_seg = imslice

        else:
            imslice = np.tile(np.expand_dims(imslice, -1), 3)
        if self.showSeg:
            imSeg = np.ones_like(self.segSlice.astype(np.float64))*10000
            imSeg = np.tile(np.expand_dims(imSeg, -1), 3)
            #imSeg[:,3]= 1
            uq = np.unique(self.segSlice.astype(np.float64))
            if 9876 not in self.colorInds:#len(self.colorsCombinations):
                selected_ud = self.colorInds
            else:
                selected_ud = uq
            for u in uq:
                if u == 0:
                    continue
                elif u == 1500:
                    color = [0.9,0.5,0.0,1]
                    ind = self.segSlice == u
                    try:
                        imSeg[ind, 0] = color[0]*255.0
                        imSeg[ind, 1] = color[1]*255.0
                        imSeg[ind, 2] = color[2]*255.0
                    except:
                        print('Please check the color index {}'.format(u))
                if u in selected_ud and u in self.colorsCombinations:
                    color = self.colorsCombinations[u]
                    ind = self.segSlice == u
                    try:
                        imSeg[ind, 0] = color[0]*255.0
                        imSeg[ind, 1] = color[1]*255.0
                        imSeg[ind, 2] = color[2]*255.0
                    except:
                        print('Please check the color index {}'.format(u))
                #imslice[ind, 3] = color[3]

            ind_total = imSeg.sum(2)!=30000
            imslice_seg = imslice
            try:
                if ind_total.sum()>0:
                    imslice_seg[ind_total, :] = cv2.addWeighted(imslice[ind_total, :].astype('uint8'), 1-self.intensitySeg, imSeg[ind_total, :].astype('uint8'), self.intensitySeg, 1)
            except Exception as e:
                print(e)
            if len(imslice)!=0:
                imslice_seg = imslice_seg.clip(0, 255)
        else:
            imslice_seg = imslice


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.imWidth, self.imHeight, 0,
                             GL_RGB, GL_UNSIGNED_BYTE,
                             imslice_seg)

        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_TEXTURE_2D)  # Disable texturing


        ##############################################################################

        #genList = glGenLists(1)
        #glNewList(1, GL_COMPILE)


    def updateEvents(self):
        """
        Updating events
        :return:
        """
        self.enabledPan = False
        #self.enabledGoTo = False
        self.enabledRotate = False
        self.enabledZoom = False
        self.enabledRuler = False
        self.points = []
        #self.setCursor(Qt.ArrowCursor)


##########################################################################################################################

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    #window = MainWindow0()
    #window.showMaximized()
    sys.exit(app.exec_())
