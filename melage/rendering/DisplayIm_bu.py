#!/usr/bin/env python
# -*- coding: utf-8 -*-

__AUTHOR__ = 'Bahram Jafrasteh'
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from melage.utils.utils import cursorPaint, zonePoint, cursorOpenHand, \
    try_disconnect, cursorErase,\
    ConvertPToPolygons, ConvertPointsToPolygons, findIndexWhiteVoxels, generate_extrapoint_on_line, PermuteProperAxis, magic_selection,permute_axis
#import OpenGL.GL as gl
from OpenGL.GL import *
import sys
import numpy as np
from skimage.segmentation import flood
from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt, QEvent
from PyQt5.QtGui import (QColor, QMatrix4x4, QOpenGLShader, QKeyEvent,
        QOpenGLShaderProgram, QPainter, QWheelEvent)
from PyQt5.QtWidgets import QOpenGLWidget, QWidget, QMenu, QAction
import cv2
import math

from collections import defaultdict
from sys import platform
if platform=='darwin':
    from melage.rendering.helpers.Shaders_120 import vsrc, fsrc, fsrcPaint, vsrcPaint
else:
    from melage.rendering.helpers.Shaders_330 import vsrc, fsrc, fsrcPaint, vsrcPaint

from melage.utils.utils import LargestCC
from sklearn.mixture import GaussianMixture

"""
import pyfftw
fft = pyfftw.interfaces.numpy_fft.fft2
ifft = pyfftw.interfaces.numpy_fft.ifft2
fftshift = pyfftw.interfaces.numpy_fft.fftshift
ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
"""
fft = np.fft.fft2
ifft = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

class GLWidget(QOpenGLWidget):
    """
    The main class to visualize image using MELAGE
    """
    ##### the list of signals used in the class #####
    zRotationChanged = pyqtSignal(int) #change rotation angle
    clicked = pyqtSignal() #
    segChanged = pyqtSignal(object, object, object, object) # segmentation changed
    LineChanged = pyqtSignal(object) # line changed
    goto = pyqtSignal(object, object) # if go to signal is activated
    zoomchanged = pyqtSignal(object, object) # zooming is activated
    rulerInfo = pyqtSignal(object, object) # ruler is activated
    sliceNChanged = pyqtSignal(object) # changing slice number
    NewPoints = pyqtSignal(object, object) # adding new points
    mousePress = pyqtSignal(object) # if mouse pressed
    interpolate = pyqtSignal(object) # if interpolation is required

    def __init__(self, colorsCombinations, parent=None, currentWidnowName = 'sagittal',
                 imdata=None, type= 'eco',id=0
                 ):
        super(GLWidget, self).__init__(parent)
        #print(self.format().version())
        #fmt =QtGui.QSurfaceFormat()
        #fmt.setVersion(1, 1)
        #fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        #self.setFormat(fmt)
        self.id = id # unique id of the class
        self.colorsCombinations = colorsCombinations # coloring scheme
        self.color_name = []
        self.imType = type # image type
        self.affine = None # image affine
        self.tract = None # tractography
        self.colorObject = [1,0,0,1] # RGBA
        self._magic_slice = None
        self.colorInd = 9876 # index of colr
        self.colorInds = [9876] # indices of colors
        self.intensitySeg = 1.0 # intensity of the segmentation
        self.N_rulerPoints = 0 # number of rulers used
        self.showAxis = False # show axis
        self.n_colors = 1 # number of colors used

        self._n_colors_previous = np.inf
        self.smooth = True # smoothing visualization
        self._threshold_image = defaultdict(list) # thresholding image
        self._allowed_goto = False
        self._selected_seg_color = 1 # segmentation color

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
        self.width_line_tract = 3 # width line for tractography
        self.polygon_info = [] # inforamtion of polygon
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
        self.enabledCircle = False # circle segmentation
        self._NenabledCircle = 1 # circle segmentation
        self._center_circle = [] # center of circles
        self._radius_circle = 5 # radius of circles
        self._tol_magic_tool = 5 #tolerance magic tool
        self._tol_cricle_tool = 1 # tolerance circle tool
        self.enabledRotate = False # Rotating
        self.enabledPen = False # Polygon Drawing
        self.enabledMagicTool = False #FreeHand Drawing
        self.enabledErase = False # Erasing the points
        self.enabledZoom = False # Zooming
        self.enabledPointSelection = False # Point Selection
        self.enabledGoTo = False # GOTO
        self.enabledRuler = False  # Ruller
        self.enabledLine = False # Line
        self.tract = None # Tractography
        self.width_line_tract = 3 # width tractography
        self._selected_seg_color = 1 # color of the segmetation
        self.points = [] # points selected
        self.penPoints = [] # pen points selected
        self.erasePoints = [] # points that surrond subject to erase it
        self.rulerPoints = defaultdict(list) # ruler points
        self.linePoints = [] # line points
        self.startLinePoints = [] # start line points
        self.guidelines_h = [] # guide lines horizontal
        self.guidelines_v = [] # guide lines vertical
        self.selectedPoints = [] # selected points
        self.colorObject = [1,0,0,1] # color object
        self.colorInd = 9876 # color index
        self.colorInds = [9876] # color indices
        self.affine = np.eye(4)

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
        self.tract = None # tractography
        self._selected_seg_color = 1 # selected segmentation color
        self._n_colors_previous = np.inf # number of colors previous
        self.width_line_tract = 3 # width line tractography
        self.hamming = False # use hamming filter
        self.widthPen = 0.0 # width pen
        self.senderPoints = None # sender points
        self.senderWindow = None # sender window
        # update image and info
        #self.updateInfo()
        self.setFocusPolicy(Qt.StrongFocus)
        self.clearColor = QColor(Qt.black)
        self.xRot = 0 # rotation in x
        self.yRot = 0 # rotation in y
        self.zRot = 0 # rotation in z
        self._threshold_image = defaultdict(list) # image thresholding

        self.enabledPan = False # Panning
        self.enabledCircle = False # circle
        self._NenabledCircle = 1
        self._center_circle = []
        #self._radius_circle = 5
        """
        
        self.enabledRotate = False # Rotating
        self.enabledPen = False # Polygon Drawing
        self.enabledMagicTool = False #FreeHand Drawing
        self.enabledErase = False # Erasing the points
        self.enabledZoom = False # ZOOM DISABLED
        self.enabledPointSelection = False # Point Selection
        self.enabledGoTo = False
        self.enabledRuler = False # Ruler
        self.enabledLine = False
        """
        self.lastPos = QPoint()
        self.points = []
        self.penPoints = []
        self.erasePoints = []
        self.rulerPoints = defaultdict(list)
        self.linePoints = []
        self.guidelines_h = []
        self.guidelines_v = []
        self.startLinePoints = []
        self.selectedPoints = []
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
        m_old =-1.0
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
        if len(shape)==3:
            self.imXMax, self.imYMax, self.imZMax = shape
        elif len(shape)== 4: #RGB
            self.imXMax, self.imYMax, self.imZMax, _ = shape

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
        self.tract = tract

        #if imdata is not None:
            #self.imdata = self.updateImage(imdata)
        #self.setNewImage.emit(shape)
        if initialState:
            self.initialState()
            self.updateCurrentImageInfo(shape)
            self.UpdatePaintInfo()

        self._kspaces = defaultdict(list)
        if self.enabledMagicTool or self.enabledCircle:
            self._magic_slice = None





        self.makeObject()

    def ShowContextMenu(self, pos):
        """
        Context Menu for the current class when panning allowed
        :param pos:
        :return:
        """
        menu = QMenu("Edit")
        remove_action = QAction("Close polygon")
        remove_action.triggered.connect(self.closePolygon)
        escape_action = QAction("Cancel")
        escape_action.triggered.connect(self.escapePolygon)

        menu.addAction(remove_action)
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

        if type(angle)!=float:
            return
        angle_action = QAction("Angle {:.2f} degrees".format(angle))
        xy_action = QAction("Center X {:.1f}, Y {:.1f}".format(pos_ruler[0], pos_ruler[1]))
        remove_action.triggered.connect(self.removeRuler)
        send_action = QAction("Send to Table")
        #send_action.triggered.connect(partial(self.sendRulerValue, [lenght, angle], 1))
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
        #send_action.triggered.connect(partial(self.sendRulerValue, [lenght, angle], 1))
        menu.addAction(gen_action)
        menu.addAction(empty_action)

        action = menu.exec_(self.mapToGlobal(pos))
        if action == gen_action:
            self.LineChanged.emit([[], [], False, True])
        elif action==empty_action:
            self.linePoints = []
            self.startLinePoints = []
            self.LineChanged.emit([[], self.colorInd, True, False])



    def ShowContextMenu_contour(self, pos):
        """
        Context Menu for contouring including center of contouring, perimeter, interpolation, etc.
        :param pos:
        :return:
        """
        from melage.utils.utils import point_in_contour
        menu = QMenu("Ruler")
        remove_action = QAction("Cancel")
        x, y = self.to_real_world(pos.x(), pos.y())
        color = self.segSlice[int(y), int(x)]
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
            centerXY = [x, y]
        self.penPoints = []
        area_action = QAction("Surface {:.2f} mm\u00b2".format(area))
        perimeter_action = QAction("Perimeter {:.2f} mm".format(perimeter))

        #xyz = [x, y, self.sliceNum, 1]
        if self.currentWidnowName == 'sagittal':
            #xyz = [xyz[1], xyz[0], xyz[2], 1]
            xyz = [self.sliceNum, centerXY[1], centerXY[0], 1]
        elif self.currentWidnowName == 'coronal':
            #xyz = [xyz[1], xyz[2], xyz[0], 1]
            xyz = [centerXY[1], self.sliceNum, centerXY[0], 1]
        elif self.currentWidnowName == 'axial':
            #xyz = [xyz[2], xyz[1], xyz[0], 1]
            xyz = [centerXY[1], centerXY[0], self.sliceNum, 1]
        loc = self.affine @ np.array(xyz)
        xy_action = QAction("Loc ({:.1f}, {:.1f},{:.1f})".format(loc[0], loc[1], loc[2]))
        if color>0:
            name_area = QAction(f"{self.color_name[color-1]}")
        else:
            name_area = QAction(f"Unknown")
        send_action = QAction("Send to Table")
        interploateadd_action = QAction("Add to interploation")
        apply_interpolation = QAction("Apply interploation")
        #send_action.triggered.connect(partial(self.sendRulerValue, [area], 0))
        remove_action.triggered.connect(self.emptyPenPoints)
        menu.addAction(name_area)
        menu.addAction(xy_action)
        menu.addAction(area_action)
        menu.addAction(perimeter_action)
        menu.addAction(send_action)
        menu.addSeparator()
        menu.addAction(interploateadd_action)
        menu.addAction(apply_interpolation)
        menu.addSeparator()
        menu.addAction(remove_action)
        action = menu.exec_(self.mapToGlobal(pos))
        if action == send_action:
            #['ImType', 'Area', 'Perimeter', 'Slice', 'WindowName', 'CenterXY']
            if area==0:
                return
            vals = []
            vals.append('{}'.format(color))

            if self.imType == 'eco':
                vals.append('Top')
            else:
                vals.append('Bottom')
            vals.append("{:.2f} mm\u00b2".format(area))
            vals.append("{:.2f} mm".format(perimeter))
            vals.append(str(self.sliceNum))
            vals.append(self.currentWidnowName)

            xyz = [x, y, self.sliceNum]
            if self.currentWidnowName == 'sagittal':
                xyz = [xyz[1], xyz[0], xyz[2]]
            elif self.currentWidnowName == 'coronal':
                xyz = [xyz[1], xyz[2], xyz[0]]
            elif self.currentWidnowName == 'axial':
                xyz = [xyz[2], xyz[1], xyz[0]]

            vals.append("{:.2f},{:.2f}".format(centerXY[0], centerXY[1]))
            vals.append('')
            self.rulerInfo.emit(vals, 0)
        elif action == interploateadd_action:
            if WI_index is not None:
                self.interpolate.emit([self.sliceNum, self.currentWidnowName, False, WI_index])
        elif action == apply_interpolation:
            if WI_index is not None:
                self.interpolate.emit([self.sliceNum, self.currentWidnowName, True, WI_index])
        return
    def emptyPenPoints(self):
        """
        eliminating pen points
        :return:
        """
        self.penPoints = []


    def sendRulerValue(self, values, column):
        # send value obtained from ruler
        if values[0] != 0:
            self.rulerInfo.emit(values, column)



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

    def closePolygon(self):
        if len(self.points) > 1:
            if len(self.points) > 2:
                try:
                    polygonC = ConvertPToPolygons(self.points)
                    for poly in polygonC:
                        whiteVoxels = np.empty((0,3))
                        if len(poly.exterior.xy[0]) > 1:
                            #print(list(poly.exterior.coords))
                            whiteVoxel, edges = findIndexWhiteVoxels(poly, self.currentWidnowName)
                            whiteVoxels = np.vstack((whiteVoxels, whiteVoxel))

                            #if whiteInd is not None:
                                #self.imSeg[tuple(zip(*whiteInd))] = self.colorInd
                    if whiteVoxels.shape[0] != 0:
                        self.segChanged.emit(whiteVoxels.astype("int"), self.currentWidnowName, self.colorInd, self.sliceNum)
                except Exception as e:
                    print('something')
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

    #def sizeHint(self):
     #   return QSize(self.imWidth, self.imHeight)

    #def rotateBy(self, xAngle, yAngle, zAngle):
     #   self.xRot = 0.0
      #  self.yRot = 0.0
       # self.zRot += -(zAngle)
        #self.update()

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
        #if self.enabledCircle:
            #self.zoomchanged.emit(self._radius_circle/abs(self.to_real_world( 1, 0)[0] - self.to_real_world(0, 0)[0]), True)
        self.zoomchanged.emit(None, True)
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

        self.makeObject()

    def UpdatePaintInfo(self):
        """
        Update information of the image with a simplified "fit" scaling.

        This centers the image and scales it to fit within the window
        while preserving its aspect ratio, creating "letterboxing" or
        "pillarboxing" as needed.
        """
        # Add a small epsilon to prevent division by zero
        epsilon = 0.000001

        window_w = self.width()
        window_h = self.height() + epsilon
        img_w = self.imWidth
        img_h = self.imHeight
        #print(f"windw w {window_w} and {window_h}")
        # Handle case where image might not be loaded yet
        if img_w == 0 or img_h == 0:
            self.left = 0
            self.right = window_w
            self.bottom = 0
            self.top = window_h
            return

        img_h += epsilon  # Add epsilon to image height as well

        # 1. Calculate Aspect Ratios
        window_ar = window_w / window_h
        image_ar = img_w / img_h

        # 2. Compare aspect ratios to determine the limiting dimension
        if window_ar > image_ar:
            # --- CASE 1: Window is "wider" than the image ---
            # The image is "taller" relative to the window.
            # We are limited by the image's HEIGHT.
            # This will create "pillarboxing" (black bars on left/right).

            # The view's height will be the image's height
            view_height = img_h
            # The view's width must be calculated from the view's height
            # to match the window's aspect ratio.
            view_width = view_height * window_ar

            # Center the view horizontally around the image's center
            self.bottom = 0
            self.top = img_h
            self.left = (img_w / 2.0) - (view_width / 2.0)
            self.right = (img_w / 2.0) + (view_width / 2.0)


        else:
            # --- CASE 2: Image is "wider" than the window ---
            # The image is "wider" relative to the window.
            # We are limited by the image's WIDTH.
            # This will create "letterboxing" (black bars on top/bottom).

            # The view's width will be the image's width
            view_width = img_w
            # The view's height must be calculated from the view's width
            # to match the window's aspect ratio.
            view_height = view_width / window_ar

            # Center the view vertically around the image's center
            self.left = 0
            self.right = img_w
            self.bottom = (img_h / 2.0) - (view_height / 2.0)
            self.top = (img_h / 2.0) + (view_height / 2.0)



        # These are just for information, not used by glOrtho
        self.zoomed_width = view_width
        self.zoomed_height = view_height

        self.halfH = self.height() // 2
        glEnable(GL_NORMALIZE)

    def drawImage(self):

        # pre requisites and drawImage
        self.drawImagePre()


        # draw additional
        self.drawImPolygon()  # draw imFreeHand


        if len(self.erasePoints) > 1:
            self.DrawerasePolygon()

        if len(self.rulerPoints.keys()) >= 1:
            self.DrawRulerLines()

        if len(self.linePoints) > 3:
            self.DrawLines(self.linePoints)

        if self.enabledCircle:
            self.DrawCricles(self._center_circle)

        if self.tract is not None: # for tract file
            self.Draw_tract(self.tract, self.width_line_tract)

        if len(self.guidelines_h) > 0:
            self.DrawLines(self.guidelines_h, self.colorh)
        if len(self.guidelines_v)>0:
            self.DrawLines(self.guidelines_v, self.colorv)

        # end iamges
        self.drawImageEnd()


    def Draw_tract(self, tract, width_line=3):
        """
        Draw tractography files
        :param tract:
        :param width_line:
        :return:
        """
        from melage.utils.utils import divide_track_to_prinicipals
        uq = np.unique(tract[:,-1])
        r = 0
        for u in uq:
            trct = tract[tract[:, -1] == u, :-1]

            if trct.shape[0]==trct[0,-1]:
                trcts = divide_track_to_prinicipals(trct[:,:3])
                colorv = [trct[0,3], trct[0,4], trct[0,5], 1]
                self.DrawLines(tuple(trcts), colorv = colorv, width_line=width_line)
                r +=1
        print(r)

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

        #version_str = glGetString(GL_VERSION)
        #print(f"OpenGL Version: {version_str.decode('utf-8')}")
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


    def draw_closed_polygon(self):
        """
        Draw polygon
        :return:
        """
        glPushMatrix()
        glPushAttrib(GL_CURRENT_BIT)
        glPolygonMode(GL_BACK, GL_LINE) # filling the polygon

        glDisable(GL_LIGHTING)
        glNormal3f(0.5, 1.0, 1.0)
        glLineWidth(5.0)
        glBegin(GL_POLYGON)  # These vertices form a closed polygon
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glVertex2f(0.4* self.imHeight, 0.2 * self.imWidth)
        glVertex2f(0.6* self.imHeight, 0.2 * self.imWidth)
        glVertex2f(0.7* self.imHeight, 0.4 * self.imWidth)

        glEnd()
        glPopAttrib()
        glPopMatrix()

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
        self.UpdatePaintInfo()
        side = min(width, height)
        #Set viewport
        glViewport((width- side) // 2, (height-side) // 2, self.imWidth,
                self.imHeight)

    def mousePressEvent(self, event):
        self.mousePress.emit(event)
        self.lastPos = event.pos()
        """
        
        if self.activateSobel:
            #event_r = QtGui.QResizeEvent(self.size(), self.size())
            #self.resizeEvent(event_r)
            self.makeCurrent()
            self.paintGL()
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            px = 20
            z = glReadPixels(self.lastPos.x()-px, self.height()-self.lastPos.y()-px, px*2+1, px*2+1, GL_RGBA, GL_BYTE)[::-1][:,:,[0,1,2]]
            z = cv2.resize(z.astype(np.uint8), (int((px*2+1)/(self.width()/self.imWidth)), int((px*2+1)/(self.height()/self.imHeight))), interpolation=cv2.INTER_AREA)
            result = np.all(z == [0, 127, 0], axis=2)

            xs, ys = np.where(result)
            x, y = self.to_real_world(event.pos().x(), event.pos().y())
            x, y = x - z.shape[0]//2, y - z.shape[1]//2
            xs, ys = np.round(y) +xs, np.round(x) + ys
            whiteVoxels = np.zeros((len(xs), 3))
            whiteVoxels[:, 1] = ys
            whiteVoxels[:, 0] = xs
            whiteVoxels[:, 2] = self.sliceNum
            self.segChanged.emit(whiteVoxels.astype("int"), self.currentWidnowName, self.colorInd)
            self.update()
        """
        if self.enabledMagicTool:
            xc, yc = event.pos().x(), event.pos().y()

            realx, realy = self.to_real_world(xc, yc)

            initial_point = (int(realx), int(realy))
            #input_point = np.expand_dims(np.array(initial_point),0)
            #self._sam_predictor.set_image(np.repeat(self.imSlice[:, :, np.newaxis], 3, axis=2))
            #input_label = np.array([1])
            #seg_new, _, _ = self._sam_predictor.predict(point_coords=input_point, point_labels=input_label,
            #                                         multimask_output=False, )
            seg_new = magic_selection(self.imSlice, initial_point, connectivity=4, tol=self._tol_magic_tool)
            self._magic_slice = None
            if seg_new is not None and self.colorInd!= 9876:
                    l1 = list(np.where(seg_new > 0))
                    whiteInd = np.stack([l1[1], l1[0], len(l1[0]) * [self.sliceNum]]).T
                    whiteInd, _ = permute_axis(whiteInd, whiteInd, self.currentWidnowName)
                    self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, self.colorInd, self.sliceNum)
                    self.makeObject()
                    self.update()



        elif self.enabledCircle:
            if self._magic_slice is not None:
                    slc = self._magic_slice.sum(2)
                    slc = slc/slc.max()
                    l1 = list(np.where((slc-self.segSlice) > 0))
                    if len(l1[0])>0:
                        whiteInd = np.stack([l1[1], l1[0], len(l1[0]) * [self.sliceNum]]).T
                        whiteInd, _ = permute_axis(whiteInd, whiteInd, self.currentWidnowName)
                        self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, self.colorInd, self.sliceNum)
                        self.makeObject()
                        self.update()

            if self._NenabledCircle>0 or 1>2:
                xc, yc = event.pos().x(), event.pos().y()
                num_segments = 50
                #if self.zoom_level_x<=1:
                radius = self._radius_circle/4#*self.imSpacing[0]#*self.zoom_level_x
                #else:
                #    radius = self._radius_circle/self.zoom_level_x
                points = []
                realx, realy = self.to_real_world(xc, yc)
                for ii in range(num_segments):
                    theta = 2.0 * np.pi * ii / num_segments  # get the current angle
                    x = radius * np.cos(theta)  # calculate the x component
                    y = radius * np.sin(theta)  # calculate the y component
                    #x1, y1 = self.to_real_world(xc+x,yc+y)
                    x1, y1 = realx +x, realy + y
                    points.append([x1,y1, self.sliceNum])
                self._center_circle = points
                self.update()
            self._NenabledCircle += 1
        elif self.enabledPointSelection:

            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.selectedPoints.append([x,y, self.sliceNum])
            whiteInd, edges = findIndexWhiteVoxels(self.selectedPoints, self.currentWidnowName, is_pixel=True,bool_permute_axis=True)
            print(whiteInd[0])
            self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, True, self.sliceNum)
            self.selectedPoints = []
            self.update()

        elif self.enabledPen:
            x, y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.penPoints.append([x, y, self.sliceNum])
        elif self.enabledRotate:
            self.lastPressPos = self.lastPos
            self.pressZone = zonePoint(event.x(), event.y(), self.width() / 2, self.width() / 2)
        elif self.enabledErase:
            self.erasePoints = []
            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.erasePoints.append([x, y, self.sliceNum])
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

        elif self.enabledLine:
            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.startLinePoints.append([x, y, self.sliceNum])

        if self.enabledGoTo:
            x , y = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.goto.emit([x, y, self.sliceNum], [self.currentWidnowName, self.imType])
            if event.button() == Qt.LeftButton:
                self._allowed_goto = True


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

    def leaveEvent(self, event):
        if self.enabledMagicTool or self.enabledCircle:
            self._magic_slice = None
            self.makeObject()
            self.update()
        super().leaveEvent(event)


    def mouseMoveEvent(self, event):
        if self.enabledMagicTool:
            xc, yc = event.pos().x(), event.pos().y()

            realx, realy = self.to_real_world(xc, yc)
            if realx<=0 or realy<=0:
                return
            initial_point = (int(realx), int(realy))


            seg_new = magic_selection(self.imSlice, initial_point, connectivity=4, tol=self._tol_magic_tool)

            if seg_new is not None and self.colorInd!= 9876:
                    color = self.colorsCombinations[self.colorInd]
                    color_image = np.stack((seg_new * color[0]*255, seg_new * color[1]*255, seg_new * color[2]*255), axis=-1)
                    self._magic_slice = color_image
                    self.makeObject()
                    self.update()
            else:
                self._magic_slice = None
        elif self.enabledCircle:
            # from MELAGE.utils.utils import
            if self._NenabledCircle > 0 and self.colorInd != 9876:

                self._center_circle = []
                xc, yc = event.pos().x(), event.pos().y()
                xc_mp, yc_mp = self.to_real_world(xc, yc)

                num_segments = 50
                # if self.zoom_level_x<=1:
                radius = self._radius_circle / 4.0  # *self.imSpacing[0]#*self.zoom_level_x
                # else:
                #    radius = self._radius_circle/self.zoom_level_x
                sum_diff = 100
                realx, realy = self.to_real_world(xc, yc)
                if hasattr(self, 'lastPos'):
                    xc_l, yc_l = self.lastPos.x(), self.lastPos.y()
                    realx_l, realy_l = self.to_real_world(xc_l, yc_l)
                    sum_diff = abs(realx_l-realx)+abs(realy_l-realy)
                if sum_diff>0.1:
                    #print(sum_diff)
                    points = []
                    for ii in range(num_segments):
                        theta = 2.0 * np.pi * ii / num_segments  # get the current angle
                        x = radius * np.cos(theta)  # calculate the x component
                        y = radius * np.sin(theta)  # calculate the y component
                        # x1, y1 = self.to_real_world(xc+x,yc+y)
                        x1, y1 = realx + x, realy + y
                        points.append([x1, y1, self.sliceNum])
                    if len(points) > 2:
                        from melage.utils.utils import seperate_lcc
                        polygonC = ConvertPointsToPolygons(points, width=0)
                        whiteInd = None
                        if polygonC.is_valid:
                            whiteInd, edges = findIndexWhiteVoxels(polygonC, self.currentWidnowName,
                                                                   bool_permute_axis=False)
                        else:
                            # print('not valid polygon')
                            # polygonC = polygonC.buffer(0)
                            multipolys = ConvertPToPolygons(self.penPoints)
                            whiteInd = np.empty((0, 3))
                            for poly in multipolys:
                                if poly.is_valid:
                                    if poly.area < 2:
                                        continue
                                    voxels, edges = findIndexWhiteVoxels(poly, self.currentWidnowName)
                                    if voxels is not None:
                                        whiteInd = np.vstack((whiteInd, voxels))

                        #center_intensity = self.imSlice[int(yc_mp), int(xc_mp)]
                        ind_1 = (whiteInd[:, 1] < self.imSlice.shape[0])*(whiteInd[:, 1]>0)
                        whiteInd = whiteInd[ind_1, :]
                        ind_2 = (whiteInd[:, 0] < self.imSlice.shape[1]) * (whiteInd[:, 0] > 0)
                        whiteInd = whiteInd[ind_2, :]
                        if len(whiteInd)==0:
                            return  super().mouseMoveEvent(event)
                        new_index_intensity = self.imSlice[whiteInd[:, 1], whiteInd[:, 0]]



                        # seg_c =  self.segSlice[int(yc_mp), int(xc_mp)]
                        segs = self.segSlice[whiteInd[:, 1], whiteInd[:, 0]]

                        # ind_seg = segs != seg_c
                        min_whit = whiteInd.min(0)
                        #from scipy.ndimage import gaussian_filter
                        a1 = whiteInd[:, [1, 0]] - min_whit[[1, 0]]
                        roi_shape = a1.max(0) + 1
                        is_rgb = self.imSlice.ndim == 3
                        if is_rgb:
                            # Initialize 3D array for RGB ROI
                            im1 = np.zeros((roi_shape[0], roi_shape[1], self.imSlice.shape[2]),
                                           dtype=self.imSlice.dtype)
                        else:
                            # Initialize 2D array for Gray ROI
                            im1 = np.zeros(roi_shape, dtype=self.imSlice.dtype)

                        #im1 = np.zeros(a1.max(0) + 1)
                        #seg1 = np.zeros_like(im1)
                        seg1 = np.zeros((roi_shape[0], roi_shape[1]), dtype=self.segSlice.dtype)
                        try:
                            im1[tuple(
                                zip(*a1))] = new_index_intensity  # im1[a1[:, 1], a1[:, 0]] = new_index_intensity
                            seg1[tuple(zip(*a1))] = segs
                        except:
                            pass

                        # We need a single channel for equalizeHist and simple flood filling
                        if is_rgb:
                            # Convert ROI to Gray for calculation purposes
                            # Assuming standard RGB (or RGBA), we ignore Alpha for calculation
                            im_for_calc = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        else:
                            im_for_calc = im1.astype(np.uint8)

                        seed_point = (int(yc_mp) - min_whit[1], int(xc_mp) - min_whit[0])

                        im_for_calc = cv2.equalizeHist(im_for_calc)
                        # im1 = cv2.GaussianBlur(im1, (0, 0), 1)
                        try:
                            # Use the enhanced grayscale image for the flood fill
                            segmented_area = flood(im_for_calc, seed_point, connectivity=1,
                                                   tolerance=self._tol_cricle_tool * im_for_calc.std())

                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            sub_seg_new = cv2.morphologyEx((segmented_area>0).astype(np.float32), cv2.MORPH_CLOSE, kernel)
                            im_l, im_f = LargestCC(sub_seg_new, 1)
                            if im_f.shape[0]>1:
                                index_sel = im_l[seed_point[1], seed_point[0]]
                                sub_seg_new = (im_l == index_sel).astype('int')
                        except:
                            sub_seg_new = None
                        if sub_seg_new is not None and self.colorInd != 9876:
                            color = self.colorsCombinations[self.colorInd]
                            seg_new = np.zeros_like(self.segSlice)
                            seg_new[whiteInd[:, 1], whiteInd[:, 0]] = sub_seg_new[tuple(zip(*a1))]
                            color_image = np.stack(
                                (seg_new * color[0] * 255, seg_new * color[1] * 255, seg_new * color[2] * 255), axis=-1)
                            self._magic_slice = color_image
                            self.makeObject()
                            self.update()
                        else:
                            self._magic_slice = None

        def compute_line(x, m, b):
            return m*x+b
        #print(self.cursor().pos())

        if self.enabledPen and len(self.penPoints)>0:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x , y  = self.to_real_world(event.pos().x(), event.pos().y())

            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.penPoints.append([x, y, self.sliceNum])
            self.updateCursor(event, x, y, dx, dy, endIm[0])

            # update the window
            self.update()

        #elif self.enabledCircle:
        #    x, y = self.to_real_world(event.pos().x(), event.pos().y())
        #    self._center_circle = [x,y]
        #    self.update()
        elif self.enabledErase  and len(self.erasePoints)>0:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x , y  = self.to_real_world(event.pos().x(), event.pos().y())

            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            self.erasePoints.append([x, y, self.sliceNum])
            self.updateCursor(event, x, y, dx, dy,endIm[0])

            # update the window
            self.update()
        elif self.enabledRuler: # if RULER is activated
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x , y  = self.to_real_world(event.pos().x(), event.pos().y())


            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
            if 'points' in self.rulerPoints[self.N_rulerPoints]:
                if len(self.rulerPoints[self.N_rulerPoints]['points'])>0:
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
        elif self.enabledLine and len(self.startLinePoints)>0:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x , y  = self.to_real_world(event.pos().x(), event.pos().y())
            endIm = self._endIMage(x, y)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]

            if len(self.linePoints)>3:
                try:
                    self.linePoints = generate_extrapoint_on_line(self.startLinePoints[0], [x, y, self.sliceNum], self.sliceNum)
                except:
                    pass
            else:
                self.linePoints.append([x, y ,self.sliceNum])

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
            self._allowed_goto = False
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

        if self.enabledGoTo and self._allowed_goto:
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
            self.enabledMagicTool = True

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
            self.enabledErase = True
            self.setCursor(cursorErase())
            try_disconnect(self)
        elif event.key() == Qt.Key_4:  # ImPaint
            self.updateEvents()
            self.enabledPen = True
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
        self._allowed_goto = False
        self.clicked.emit()
        if self.enabledErase:
            # erase mode
            if len(self.erasePoints)>2:
                self.erasePoints.append(self.erasePoints[0])
                polErase = ConvertPToPolygons(self.erasePoints) # convert to polygons
                self.erasePolygon(polErase)
        elif self.enabledCircle:
            #from MELAGE.utils.utils import
            if self._NenabledCircle>0:

                self._center_circle=[]
                xc, yc = event.pos().x(), event.pos().y()
                xc_mp, yc_mp = self.to_real_world(xc, yc)
                num_segments = 50
                #if self.zoom_level_x<=1:
                radius = self._radius_circle/4.0#*self.imSpacing[0]#*self.zoom_level_x
                #else:
                #    radius = self._radius_circle/self.zoom_level_x
                realx, realy = self.to_real_world(xc, yc)

                points = []
                for ii in range(num_segments):
                    theta = 2.0 * np.pi * ii / num_segments  # get the current angle
                    x = radius * np.cos(theta)  # calculate the x component
                    y = radius * np.sin(theta)  # calculate the y component
                    #x1, y1 = self.to_real_world(xc+x,yc+y)
                    x1, y1 = realx + x, realy + y
                    points.append([x1,y1, self.sliceNum])
                if len(points) > 2:
                    from melage.utils.utils import seperate_lcc
                    polygonC = ConvertPointsToPolygons(points, width = 0)
                    try:
                        whiteInd = None
                        if polygonC.is_valid:
                            whiteInd, edges = findIndexWhiteVoxels(polygonC, self.currentWidnowName, bool_permute_axis=False)
                        else:
                            #print('not valid polygon')
                            #polygonC = polygonC.buffer(0)
                            multipolys = ConvertPToPolygons(self.penPoints)
                            whiteInd = np.empty((0, 3))
                            for poly in multipolys:
                                if poly.is_valid:
                                    if poly.area<2:
                                        continue
                                    voxels, edges = findIndexWhiteVoxels(poly, self.currentWidnowName)
                                    if voxels is not None:
                                        whiteInd = np.vstack((whiteInd, voxels))
                            #if whiteInd is not None:
                             #   self.imSeg[tuple(zip(*whiteInd))] = self.colorInd
                        #if whiteInd is not None:
                         #   self.segSlice[tuple(zip(*whiteInd))] = self.colorInd
                        center_intensity = self.imSlice[int(yc_mp), int(xc_mp)]


                        whiteInd = whiteInd[whiteInd[:, 1] < self.imSlice.shape[0], :]
                        whiteInd = whiteInd[whiteInd[:, 0] < self.imSlice.shape[1], :]
                        new_index_intensity = self.imSlice[whiteInd[:, 1], whiteInd[:, 0]]

                        std_strategy = False
                        if std_strategy:
                            std_d = new_index_intensity.std()
                            index_proximity = (new_index_intensity > (center_intensity - 20*std_d)) * (new_index_intensity < (
                                    center_intensity + 20*std_d))
                            whiteInd = whiteInd[index_proximity, :]
                            try:
                                whiteInd = seperate_lcc(whiteInd, [int(yc_mp), int(xc_mp)])
                            except:
                                pass
                        else:
                            from skimage.segmentation import flood
                            #seg_c =  self.segSlice[int(yc_mp), int(xc_mp)]
                            segs =self.segSlice[whiteInd[:, 1], whiteInd[:, 0]]
                            #ind_seg = segs != seg_c
                            min_whit = whiteInd.min(0)
                            from scipy.ndimage import gaussian_filter
                            a1 = whiteInd[:,[1,0]] - min_whit[[1,0]]
                            im1 = np.zeros(a1.max(0) + 1)
                            seg1 = np.zeros_like(im1)
                            try:
                                im1[tuple(zip(*a1))] = new_index_intensity# im1[a1[:, 1], a1[:, 0]] = new_index_intensity
                                seg1[tuple(zip(*a1))] = segs
                            except:
                                pass
                            debug = False
                            if debug:
                                ims = np.zeros_like(self.imSlice)
                                ims[tuple(zip(*whiteInd[:, [1, 0]]))] = 0
                                ims[int(yc_mp), int(xc_mp)] = 1000
                                a1 = whiteInd[:, [1, 0]] - min_whit[[1, 0]]
                                im1 = np.zeros(a1.max(0) + 1)
                                im1[tuple(zip(*a1))] = new_index_intensity

                                ims = self.imSlice.copy()
                                ims[tuple(zip(*whiteInd[:, [1, 0]]))] = 0
                                ims[int(yc_mp), int(xc_mp)] = 1000
                                #whit = np.argwhere(ind_sel)[:, [1, 0]] + min_whit[[1, 0]]
                                #ims[tuple(zip(*whit))] = 0

                                #import matplotlib.pyplot as plt
                                #plt.imshow(ims)
                                #plt.show()

                            seed_point = (int(yc_mp) - min_whit[1], int(xc_mp) - min_whit[0])

                            im1 = cv2.equalizeHist(im1.astype(np.uint8))
                            #im1 = cv2.GaussianBlur(im1, (0, 0), 1)

                            ind_sel = flood(im1, seed_point, connectivity=1,
                                      tolerance=self._tol_cricle_tool*im1.std())*im1
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            ind_sel = cv2.morphologyEx((ind_sel > 0).astype(np.float32), cv2.MORPH_CLOSE,
                                                           kernel)

                            im_l, im_f = LargestCC(ind_sel, 1)
                            if im_f.shape[0] > 1:
                                index_sel = im_l[seed_point[1], seed_point[0]]
                                ind_sel = (im_l == index_sel).astype('int')

                            whit = np.argwhere(ind_sel)[:, [0, 1]] + min_whit[[1, 0]]
                            whiteInd = np.c_[whit[:,[1,0]], np.full(whit.shape[0], whiteInd[0, 2])]
                            #try:
                            #    whiteInd = seperate_lcc(whiteInd, [int(yc_mp), int(xc_mp)])
                            #except:
                            #    pass
                            """
                            ind_seg = seg1 == seg_c
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                            _, labels, centers = cv2.kmeans(np.float32(im1.reshape((-1, 2))), 2, None, criteria, 2,
                                                            cv2.KMEANS_RANDOM_CENTERS)
                            segmented_image = centers[labels.flatten()]
                            segmented_image = segmented_image.reshape(im1.shape)
                            seg_c = segmented_image[int(yc_mp)-min_whit[0], int(xc_mp)-min_whit[1]]
                            (segmented_image-seg_c)

                            from MELAGE.utils.utils import Threshold_MultiOtsu
                            num_class = 3

                            ths = Threshold_MultiOtsu(im1, num_class)

                            for i in range(3):
                                if i == 0:
                                    im1[im1 <= ths[i]] = i
                                elif i == len(ths):
                                    im1[im1 >= ths[i - 1]] = i
                                    print(i)
                                else:
                                    im1[(im1 <= ths[i]) * (im1 > ths[i - 1])] = i

                            
                            prob = np.exp(-(im1 - im1[ind_seg].mean()) ** 2)
                            prob = gaussian_filter(prob, sigma=2)
                            ind_sel = prob > prob.mean()
                            whit = np.argwhere(ind_sel)[:,[1,0]]+min_whit[[1,0]]
                            whiteInd = np.c_[whit, np.full(whit.shape[0], whiteInd[0, 2])]


                            prob = np.exp(-(new_index_intensity - new_index_intensity[~ind_seg].mean()) ** 2)
                            ind_sel = prob>0.5
                            whiteInd = whiteInd[ind_sel, :]
                            
                            
                            segs[ind_seg] = 0
                            segs[~ind_seg]=1
                            #X = np.concatenate([np.argwhere(ind_seg), self.imSlice[ind_seg].reshape(-1, 1)], 1)
                            X = np.concatenate([whiteInd[:, [1, 0]]], 1)
                            theta, residuals, rank, s = np.linalg.lstsq(
                                X,segs, rcond=None)
                            pred = np.dot(X, theta)
                            X_new = np.concatenate([whiteInd[:, [1, 0]]], 1)
                            pred_new = np.dot(X_new, theta).squeeze()
                            tol = 0.5
                            ind_sel = (pred_new<=1+tol)*(pred_new>=1-tol)
                            whiteInd = whiteInd[ind_sel,:]
                            """
                        whiteInd, _ = permute_axis(whiteInd, edges, self.currentWidnowName)
                        self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, self.colorInd, self.sliceNum)
                        self.polygon_info.append([polygonC.centroid.x, polygonC.centroid.y, polygonC.area])
                        self.update()
                    except Exception as e:
                        print(e)



        elif self.enabledRuler: # if ruler option is activated
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
        elif self.enabledLine: # draw line activated
            if len(self.linePoints)<=3:
                self.linePoints = []
                self.startLinePoints = []
                pass
            try:
                whiteInd = np.round(generate_extrapoint_on_line(self.startLinePoints[0], self.linePoints[-1], self.sliceNum))
                whiteInd = PermuteProperAxis(whiteInd, self.currentWidnowName)
                self.LineChanged.emit([[whiteInd[0,:],whiteInd[-1,:]], self.colorInd, False, False])
                #whiteInd= np.vstack([xs, ys, [self.sliceNum]*len(xs)]).T
                self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, 1500, self.sliceNum)
                self.update()

            except Exception as e:
                pass

            self.linePoints = []
            self.startLinePoints = []

        elif self.enabledPen: #drawing countour
            if len(self.penPoints) > 2:
                polygonC = ConvertPointsToPolygons(self.penPoints, width = self.widthPen)
                try:
                    whiteInd = None
                    if polygonC.is_valid:
                        whiteInd, edges = findIndexWhiteVoxels(polygonC, self.currentWidnowName)

                    else:
                        #print('not valid polygon')
                        #polygonC = polygonC.buffer(0)
                        multipolys = ConvertPToPolygons(self.penPoints)
                        whiteInd = np.empty((0, 3))
                        for poly in multipolys:
                            if poly.is_valid:
                                if poly.area<2:
                                    continue
                                voxels, edges = findIndexWhiteVoxels(poly, self.currentWidnowName)
                                if voxels is not None:
                                    whiteInd = np.vstack((whiteInd, voxels))
                        #if whiteInd is not None:
                         #   self.imSeg[tuple(zip(*whiteInd))] = self.colorInd
                    #if whiteInd is not None:
                     #   self.segSlice[tuple(zip(*whiteInd))] = self.colorInd
                    self.segChanged.emit(whiteInd.astype("int"), self.currentWidnowName, self.colorInd, self.sliceNum)
                    self.polygon_info.append([polygonC.centroid.x, polygonC.centroid.y, polygonC.area])
                    self.update()
                except Exception as e:
                    print(e)
            self.penPoints = []
        #elif self.enabledGoTo:



    def erasePolygon(self,polsErase):
        """
        Erase polygons by doing intersection
        :param polsErase:
        :return:
        """

        #changedKeys = []
        #points = []
        whiteVoxels = np.empty((0, 3))
        for polErase in polsErase:
            #points += fillInsidePol(polErase)
            if len(polErase.exterior.xy[0]) > 1:
                # print(list(poly.exterior.coords))
                try:
                    whiteVoxel, edges = findIndexWhiteVoxels(polErase, self.currentWidnowName)
                    whiteVoxels = np.vstack((whiteVoxels, whiteVoxel))
                except Exception as e:
                    print(e)

                # if whiteInd is not None:
                # self.imSeg[tuple(zip(*whiteInd))] = self.colorInd
        if whiteVoxels.shape[0] != 0:
            self.segChanged.emit(whiteVoxels.astype("int"), self.currentWidnowName, 0, self.sliceNum)
       # if len(points)>1:
        #    points = np.array(points).astype("int")[:,0:2][:,[1,0]]
            #segSlice = self.imSeg
            #segSlice[tuple(zip(*points))] = 0
            #self.updateSeg(segSlice)
            #self.segSlice[tuple(zip(*points))] = 0


        self.erasePoints = []


        self.update()

    def currentSegSlice(self):
        """
        Get current slice image
        :return:
        """
        if self.activeDim == 0:
            segSlice = self.imSeg[self.sliceNum, :,: ]
        elif self.activeDim == 1:
            segSlice = self.imSeg[:, self.sliceNum, :]
        elif self.activeDim == 2:
            segSlice = self.imSeg[:, :, self.sliceNum]
        return segSlice.astype(np.uint8)

    def updateSeg(self, segSlice):
        """
        Update segmented image
        :param segSlice:
        :return:
        """
        if self.activeDim == 0:
            self.imSeg[self.sliceNum, :,: ] = segSlice
        elif self.activeDim == 1:
            self.imSeg[:, self.sliceNum, :] = segSlice
        elif self.activeDim == 2:
            self.imSeg[:, :, self.sliceNum] = segSlice


    """
    
    def contourShow(self, contours):
        def toAxis(contour):
            return np.squeeze(contour)

        for contour in contours:
            contour = np.squeeze(contour)
            glBegin(GL_LINE_STRIP)
            # These vertices form a closed polygon
            print(contour)
            # glColor3f(*self.colorObject)  # Red
            # listPts = list(polyg.exterior.coords)
            if len(contour.shape)<=1:
                return
            for (x, y) in contour:
                glVertex2f(x, y)
            glVertex2f(contour[0][0],contour[0][1])
            glEnd()
    """

    def drawImPolygon(self): # draw polygons
        def changeAccordingToWN(points, windowname):
            if windowname == 'coronal':
                if self.currentWidnowName == 'sagittal':
                    points = np.array(points)[:,[2,0,1]]
                    return points
                if self.currentWidnowName == 'axial':
                    points = np.array(points)[:, [2, 0, 1]]
                    return points




        #glPushMatrix()
        glPushAttrib(GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)


        glUseProgram(self.program[1].programId())
        glUniformMatrix4fv(self.program[1].vertModelViewAttrId, 1, GL_FALSE, self.mvpMatrix)



        self.program[1].enableAttributeArray(self.program[1].vertTexCoordAttrId)
        self.program[1].setAttributeArray(self.program[1].vertTexCoordAttrId, self.coord)
        self.program[1].enableAttributeArray(self.program[1].vertPosAttrId)
        self.program[1].setAttributeArray(self.program[1].vertPosAttrId, self.vertex)

        # set brightness/contrast

        # self.deinterlace = [100, 0, 0]
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


        if len(self.penPoints)>2:
            #self.NewPoints.emit(self.penPoints, None)
            polygonC = ConvertPointsToPolygons(self.penPoints, width=self.widthPen)
            penPoints = list(polygonC.exterior.coords)
            glLineWidth(1)
            glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
            #glColor3f(1.0, 0.0, 0.0)
            for (x, y, z) in penPoints:
                glVertex2f(x, y)
            glEnd()
            glLineWidth(1.0)

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


    def DrawerasePolygon(self):
        """
        Draw polygon to erase area
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
        glUniform4fv(color_location, 1, [1,0,0, 1])

        glDisable(GL_LIGHTING)
        glNormal3f(0.5, 1.0, 1.0)
        glLineWidth(1.0)

        glBegin( GL_LINE_STRIP )  # These vertices form a closed polygon
        glColor3f(1.0, 1.0, 0.0)  # Red
        for (x, y, z) in self.erasePoints:
            glVertex2f(x, y)
        glEnd()

        glUseProgram(0)

        glPopAttrib()
        glPopMatrix()


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


    def DrawCricles(self, points):
        """
        Draw circle to segment image
        :param points:
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
        glUniform4fv(color_location, 1, [1,1,0, 1])
        #glLineStipple(1, 0x00FF)  # [1] dashed lines
        glEnable(GL_LINE_STIPPLE)
        #glDisable(GL_LIGHTING)
        glEnable(GL_LIGHTING)
        glNormal3f(0.5, 1.0, 1.0)
        glLineWidth(2.0)

        glBegin( GL_LINE_LOOP )  # These vertices form a closed polygon
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
        imslice = self.imSlice.copy()
        if activate_kspace:
            if self.sliceNum in self._kspaces.keys():
                kspace = np.copy(self._kspaces[self.sliceNum])
            else:
                kspace = fftshift(fft(ifftshift(imslice)))
                self._kspaces[self.sliceNum] = np.copy(kspace)


            # band pass filter
            if self.BandPR1 > 0:
                from melage.utils.utils import computeAnisotropyElipse
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
                a, b = np.floor(np.array((rows, cols)) / 2).astype("int")
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
            if imslice.ndim==2:
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

            ind_total = imSeg.sum(axis=2) != 30000
            try:
                imslice_seg = imslice
                if self._magic_slice is not None:
                    seg = self._magic_slice
                    imSeg[seg > 0] = seg[seg > 0]
                    ind_total = imSeg.sum(axis=2) != 30000

                if ind_total.sum()>0:

                    imslice_seg[ind_total, :] = cv2.addWeighted(imslice[ind_total, :].astype('uint8'), 1-self.intensitySeg,
                                                                imSeg[ind_total, :].astype('uint8'), self.intensitySeg, 1)

                    #ind_total = (seg-self.segSlice>0) > 0
                    #if ind_total.sum()>0:
                    #    imslice_seg[ind_total, :] = cv2.addWeighted(imslice[ind_total, :].astype('uint8'), 1 - self.intensitySeg,
                    #                              self._magic_slice[ind_total, :].astype('uint8'), self.intensitySeg, 1)
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
        self.enabledCircle = False
        self._NenabledCircle = 1
        self._center_circle= []

        self.enabledPointSelection = False
        self._allowed_goto = False
        self.enabledRotate = False
        self.enabledMagicTool = False
        self.enabledErase = False
        self.enabledZoom = False
        self.enabledRuler = False
        self.enabledLine = False
        self.erasePoints = []
        self.points = []
        self.penPoints = []
        #self.setCursor(Qt.ArrowCursor)


##########################################################################################################################

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    #window = MainWindow0()
    #window.showMaximized()
    sys.exit(app.exec_())
