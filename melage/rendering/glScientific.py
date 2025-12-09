# -*- coding: utf-8 -*-
__author__= 'Bahram Jafrasteh'
import numpy as np
import sys
sys.path.append('../')
from melage.rendering import GLViewWidget, GLAxisItem, GLScatterPlotItem, GLGridItem, GLVolumeItem, GLPolygonItem
from OpenGL.GL import *
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QAction, QSlider, QFileDialog
from PyQt5.QtGui import QVector3D, QMatrix4x4
from PyQt5.QtCore import Qt
import cv2
from scipy.ndimage import gaussian_filter

#from utils.polygontessellator import PolygonTessellator
from melage.utils.utils import ConvertPointsToPolygons, ConvertPToPolygons, fillInsidePol, Polygon, LargestCC
from PyQt5.QtCore import pyqtSignal
from functools import partial

class glScientific(GLViewWidget):
    """

    """
    point3dpos=pyqtSignal(object, object) #qt signal
    update_3dview = pyqtSignal(object, object)  # qt signal
    update_cmap = pyqtSignal(object, bool)

    def __init__(self, colorsCombinations, parent = None, id =0, source_directory=''):
        """

        :param colorsCombinations: a dictionary containing color information
        :param parent: parent
        :param id: id of the GL
        :param source_directory: source folder
        """

        super( glScientific, self).__init__(parent)
        self.traces = dict()
        #self = gl.GLViewWidget()
        #self.installEventFilter()
        self.id = id
        self._lastZ = 1
        #self._image = None
        self._renderMode = 'Img'
        self._threshold = 0
        self._artistic = False
        self.initiate_actions()
        self._updatePaint = True
        self._gotoLoc = False
        self.colorsCombinations = colorsCombinations
        self.totalPolyItems = []
        self.maxXYZ = [100, 100, 100]
        self._excluded_inds = np.zeros((100,100,100), bool)
        self.setWindowTitle('3D view')
        self._enabledPolygon = False
        #self.setGeometry(0, 110, 600, 600)
        self.setGeometry(0,0,1000,1000)
        self.el_az_dis.connect(
            lambda points : self.emit_viewport(points)
        )
        self._indices = None
        self._rendered = 'seg'
        self._numPolys = 0
        self.totalpolys = defaultdict(list)
        self.totalpolys_camer = defaultdict(list)

        self._sPoints = []
        self.intensityImg = 0.1
        self.intensitySeg = 1.0
        self.totalItems = defaultdict(list)
        self.totalItems['axis'] = []
        self.totalItems['polys'] = []
        self.colorInd = 9876#len(self.colorsCombinations)
        self.colorInds= [9876]
        self.setContextMenuPolicy(Qt.CustomContextMenu) # enable right click

        self.customContextMenuRequested.connect(self.ShowContextMenu)

        self.createGridAxis(self.maxXYZ)
        self.source_dir = source_directory
        self.GLV = GLVolumeItem(sliceDensity=1, smooth=True, glOptions='translucent',
                                intensity_seg = self.intensitySeg)
        self.GLSC = GLScatterPlotItem(pxMode=True, size=5)
        self.GLPl = GLPolygonItem()
        #self.GLV.smooth = True
        self._UseScatter = False

        self._verticalSlider_1 = QSlider(self)
        self._verticalSlider_1.setOrientation(Qt.Vertical)
        self._verticalSlider_1.setObjectName("verticalSlider_6")
        self._verticalSlider_1.setRange(1, 50)
        self._verticalSlider_1.setValue(1)
        self._verticalSlider_1.setVisible(False)
        #self.label_1 = QLabel(self)
        #self.label_1.setAlignment(Qt.AlignCenter)
        #self.label_1.setObjectName("label_1")
        #self._verticalSlider_1.valueChanged.connect(self.label_1.setNum)
        self._verticalSlider_1.valueChanged.connect(self.threshold_change)

        self._seg_im = None

        #self.opts['bgcolor'] = [0.5, 1, 0.0, 1]
        self.opts['bgcolor'] = [0.05, 0.05, 0.05, 1]#[0.3, 0.3, 0.3, 1] # background color

    def threshold_change(self, value):
        """
        Threshold in 3D rendering
        :param value:
        :return:
        """
        if self._rendered != 'image':
            self._verticalSlider_1.setValue(1)
            return
        self._threshold = value
        self.update_cmap_image(self._lastmap_type,reset=False)
        #self.cmap_image(self._lastmap_type,reset=False)




    def take_screenshot(self):
        """
        Take screenshot from 3D rendering
        :return:
        """
        if self._seg_im is None:
            return
        filters = "png(*.png)"
        opts = QFileDialog.DontUseNativeDialog
        fileObj = QFileDialog.getSaveFileName(self, "Open File", self.source_dir, filters, options=opts)
        if fileObj[0] == '':
            return
        filename = fileObj[0] + '.png'
        init_width, init_height = self.width(), self.height()
        maxhw = max(init_width, init_height)
        width, height = init_width, init_height
        while maxhw <3000:
            width, height = width*1.5, height*1.5
            maxhw = max(width, height)
        self.setFixedWidth(int(width))
        self.setFixedHeight(int(height))
        maxhw = max(self.width(), self.height())
        #self.setGeometry()
        self.removeItem('scatter_total')
        self.GLV.setData(self._seg_im, self._artistic)
        self.GLV.setDepthValue(10)
        self.addItem(self.GLV, 'vol_total')
        self.showEvent(0)
        maxhw = max(self.width(), self.height())
        img = glReadPixels(0, 0, maxhw, maxhw, GL_RGB, GL_FLOAT)
        img = img[::-1]
        img = img[:self.height(), :self.width(), :]
        from matplotlib.image import imsave

        imsave(filename, img)
        self.setFixedWidth(init_width)
        self.setFixedHeight(init_height)

    def initiate_actions(self):
        """

        :return:
        """
        self.axis_action = QAction("Axis")
        self.axis_action.setCheckable(True)
        self.axis_action.setChecked(False)
        self.axis_action.triggered.connect(self.axis_status)


        self.goto_action = QAction("GoTo")
        self.goto_action.setCheckable(True)
        self.goto_action.setChecked(False)
        self.goto_action.triggered.connect(self.goto_status)

        self.grid_action = QAction("Grid")
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(False)
        self.grid_action.triggered.connect(self.grid_status)



        self.seg_action = QAction("Show Seg.")
        self.seg_action.setCheckable(True)
        self.seg_action.setChecked(False)
        self.seg_action.triggered.connect(self.seg_status)

        self.im_seg_action = QAction("Show Img+Seg")
        self.im_seg_action.setCheckable(True)
        self.im_seg_action.setChecked(False)
        self.im_seg_action.triggered.connect(self.im_seg_status)



        self.screenshot_action = QAction("ScreenShot")
        self.screenshot_action.triggered.connect(self.take_screenshot)

        self.draw_action = QAction("Draw")
        self.draw_action.setCheckable(True)
        self.draw_action.setChecked(False)
        self.draw_action.triggered.connect(self.draw_status)


        #self.draw_art = QAction("Artistic")
        #self.draw_art.setCheckable(True)
        #self.draw_art.setChecked(False)
        #self.draw_art.triggered.connect(self.draw_art_action)

        self.clear_action = QAction("Show Total")
        self.clear_action.triggered.connect(self.showTotal)



    def showTotal(self):
        """
        Show all the part of image in 3D
        :return:
        """
        self._excluded_inds[:] = False
        self._artistic = False
        self.update_3dview.emit(True, None)


    def remove_painting(self):
        """
        Excluding 3D ares from demonstration
        :return:
        """
        def filledPoly(poly):
            """
            Fill inside polygons
            Args:
                poly:

            Returns:

            """
            from matplotlib.path import Path

            try:
                coords = np.array(poly.exterior.coords)

                p = Path(coords)
                xmin, ymin, xmax, ymax = poly.bounds
                x = np.arange(np.floor(xmin), np.ceil(xmax), 1)
                y = np.arange(np.floor(ymin), np.ceil(ymax), 1)
                points = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
                ind_points = p.contains_points(points)
                selected_points = points[ind_points]

                return selected_points
            except:
                return []
        def adjustVals(selected):
            for i in [0,1,2]:
                ind = selected[:,i]>=self.maxXYZ[i]
                selected[ind, i] = self.maxXYZ[i]-1
                ind = selected[:,i]<=0
                selected[ind, i] = 1
            return selected
        def findIndices(vls):
            pl = Polygon(vls)
            if not pl.is_valid:
                pls = ConvertPToPolygons(vls)
                for ij, pl in enumerate(pls):
                    if ij == 0:
                        #pl = spline_points(pl)
                        selected_points = filledPoly(pl)
                    else:
                        #pl = spline_points(pl)
                        n_points = filledPoly(pl)
                        selected_points = np.vstack([selected_points, n_points])
            else:
                #pl = spline_points(pl)
                selected_points = filledPoly(pl)
            return selected_points
        if len(self.totalpolys[0])<5:
            return
        if 'vol_total' in self.items:
            self.SubUpdateSegScatterItem()
        else:
            return
        #points = self._compute_coordinates(self.mousePos)
        if 'scatter_total' in self.items:

            self.removeItem('scatter_total')
            self.GLV.setData(self._seg_im, self._artistic)
            self.GLV.setDepthValue(20)
            self.addItem(self.GLV, 'vol_total')

        invM = np.linalg.inv(self.mvpProj.reshape((4, 4)).transpose())
        xmax, ymax, zmax = self.maxCoord
        filled_polygon = np.array(self._sPoints)
        filled_polygon = findIndices(filled_polygon)
        filled_polygon = np.concatenate([filled_polygon, np.ones((len(filled_polygon), 2))], 1)
        filled_polygon[:, 0] = 2 * filled_polygon[:,0] / self.width() - 1
        filled_polygon[:, 1] = 1 - 2 * filled_polygon[:,1] / self.height()
        filled_polygon[:, 3] = 1
        filled_polygon[:, 2] = 2 * 1- 1



        Tpoints = []
        #self._z[self._z>0].min(), #self._z[self._z<1].max()
        opt=1e-4
        #round(self._z[self._z < 1].max() + 0.0001, 4)
        for z in np.arange(0.9985- 0.0006, round(self._z[self._z < 1].max() + 0.0001, 4), 1e-6):
            A = filled_polygon.copy()
            A[:, 2] = 2 * z - 1

            Tpoints = np.matmul(invM,
                                A.T).T
            Tpoints /= Tpoints[:, 3, None]
            # Tpoints = Tpoints.astype('int')

            selected = Tpoints.astype('int')[:, :3]


            selected[:, 2] = self.maxXYZ[0] - selected[:, 2]
            selected[:, 0] = self.maxXYZ[2] - selected[:, 0]

            ind = 0
            for j, xyzm in enumerate([zmax, ymax, xmax]):
                ind += ((selected[:, j] < xyzm) * (selected[:, j] >= 0))
            ind = ind >= 3
            selected = selected[ind, :]
            selected = selected[:, [2, 1, 0]]

            self._excluded_inds[selected[:, 0], selected[:, 1], selected[:, 2]] = True


        #from scipy.ndimage import binary_fill_holes
        #self._excluded_inds = binary_fill_holes(self._excluded_inds)
        self._indices = (self._indices.astype('int')- self._excluded_inds.astype('int'))>0
        self._seg_im[self._excluded_inds, :] = 0

        #print('removing')
        self.clear_points()
        self.GLV.setData(self._seg_im, self._artistic)
        self.GLV.update()
        return


    def clear_points(self):
        """
        Clear information
        :return:
        """
        self.totalpolys = defaultdict(list)
        self.totalpolys_camer = defaultdict(list)
        self._numPolys = 0
        self._lastZ=1
        self.GLPl.update_points(self.totalpolys)
        self.update()


    def paintGL(self, region=None, viewport=None, useItemNames=False):
        """
        Paint GL
        :param region:
        :param viewport:
        :param useItemNames:
        :return:
        """
        super(glScientific, self).paintGL()
        elev = self.opts['elevation'] #* np.pi/180.
        azim = self.opts['azimuth'] #* np.pi/180.
        #print(elev, azim)

    def _endIMage(self, x, y, z, tol=10): # to check if we are at the end of the image
        zmax, xmax, ymax = self.maxCoord
        zmax, xmax, ymax = zmax+tol, ymax+tol, xmax+tol
        endIm = False
        if x<-tol:
            endIm = True
            x = -tol
        elif x>xmax:
            endIm = True
            x = xmax
        if y <-tol:
            endIm = True
            y = -tol
        elif y > ymax:
            endIm = True
            y = ymax
        if z <-tol:
            endIm = True
            z = -tol
        elif z > zmax:
            endIm = True
            z = zmax
        return endIm, x, y, z

    def mouseReleaseEvent(self, a0) -> None:
        """
        Mouse events
        :param a0:
        :return:
        """
        if self._enabledPolygon:
            if len(self._sPoints)==0:
                return
            self.totalpolys[self._numPolys].append(self.totalpolys[self._numPolys][0])
            self._sPoints.append(self._sPoints[0])

            if self.GLV._ax==0:
                self.totalpolys_camer[self._numPolys] = ['sagittal', self.GLV._d]
            elif self.GLV._ax == 1:
                self.totalpolys_camer[self._numPolys] = ['coronal', self.GLV._d]
            elif self.GLV._ax == 2:
                self.totalpolys_camer[self._numPolys] = ['axial', self.GLV._d]
            self._numPolys = 0
            self.GLPl.update_points(self.totalpolys)

            #polygonC = ConvertPointsToPolygons(self.totalpolys[self._numPolys], width=0)
            #if polygonC.is_valid:
            #    Mpl = ConvertPToPolygons(self.totalpolys[self._numPolys])
            try:
                self.remove_painting()
            except Exception as e:
                print(e)
            self._sPoints = []
            self._enabledPolygon = False
            self.draw_action.setChecked(False)


        if 'scatter_total' in self.items:

            self.removeItem('scatter_total')
            self.GLV.setData(self._seg_im, self._artistic)
            self.GLV.setDepthValue(20)
            self.addItem(self.GLV, 'vol_total')

    def _custom_viewMatrix(self, distance):
        tr = QMatrix4x4()
        tr.translate( 0.0, 0.0, -distance)
        tr.rotate(self.opts['elevation']-90, 1, 0, 0)
        tr.rotate(self.opts['azimuth']+90, 0, 0, -1)
        center = self.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr

    def _custom_projectionMatrix(self, dist):

        dpr = self.devicePixelRatio()
        region = (0, 0, self.width() * dpr, self.height() * dpr)

        x0, y0, w, h = self.getViewport()

        nearClip = dist * 0.001
        farClip = dist * 1000.

        r = nearClip * np.tan(self.opts['fov'] * 0.5 * np.pi / 180.)
        t = r * h / w

        ## Note that X0 and width in these equations must be the values used in viewport
        left = r * ((region[0] - x0) * (2.0 / w) - 1)
        right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
        bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
        top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)

        tr = QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr

    def _compute_coordinates(self, ev, z=None):
        if z is None:
            self.showEvent(0)
            z = glReadPixels(ev.x(), self.height() - ev.y(), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
        if z == 0:
            return [0,0,0,0]
        if z==1:
            z=self._lastZ
            #a = np.argwhere(self._z != 1)
            #_y, _x = self.height() - ev.y(), ev.x()
            #ind = np.abs(a-np.array([_y, _x])).sum(-1).argmin()
            #z = self._z[a[ind][0], a[ind][1]]
        self._lastZ = z



        #per = np.array(self.m.copyDataTo()).reshape((4, 4)).transpose().flatten()
        #zFar = 0.5 * per[14] * (1.0 - ((per[10] - 1.0) / (per[10] + 1.0)))
        # zNear = zFar * (per[10] + 1.0) / (per[10] - 1.0)

        # z = (2.0 * z) - 1.0
        # z = (2.0 * zNear * zFar) / (zFar + zNear - (z * (zFar - zNear)))

        ndcx = 2 * self.mousePos.x() / self.width() - 1
        ndcy = 1 - 2 * self.mousePos.y() / self.height()
        a= [self.mousePos.x(), self.mousePos.y()]

        #z=0.998

        invM = np.linalg.inv(self.mvpProj.reshape((4, 4)).transpose())

        points = np.matmul(invM,
                           np.array([ndcx, ndcy, 2 * z - 1, 1]))
        points /= points[3]
        return points,a

    def mouseMoveEvent(self, ev):
        #print(self.opts['azimuth'], self.opts['elevation'])
        if not hasattr(self, 'mousePos'):
            return
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        if self._enabledPolygon:

            #self.SubUpdateSegScatterItem()

            #z = self._z[self.height()-ev.y(), ev.x()]
            z=1.0
            points, _ = self._compute_coordinates(ev, z=z)
            #points[0] -= 1
            #points[-1] -= 1
            # points[1] -= 1
            #points[2] = self.maxXYZ[2] - points[2]
            #points[0] = self.maxXYZ[2] - points[0]
            p1, xo = self._compute_coordinates(ev, 1)

            # self.points = []
            x,y, z, _ = points
            if z == 0:
                return
            endIm = self._endIMage(x, y, z)

            if endIm[0]:
                # update x and y
                #return
                x = endIm[1]
                y = endIm[2]
                z = endIm[3]
            endIm2 = self._endIMage(p1[0], p1[2], p1[3])
            self._sPoints.append(xo)
            #if not endIm2[0]:
            #    self._sPoints.append(p1)
            self.totalpolys[self._numPolys].append([x, y, z])
            if len(self.totalpolys[self._numPolys]) > 2:
                self.GLPl.update_points(self.totalpolys)
                self.addItem(self.GLPl, 'pol_total')
                self.GLPl.setDepthValue(100)
            #if 'scatter_total' in self.items:
            #    self.removeItem('scatter_total')
            #    self.GLV.setData(self._seg_im)
            #    self.GLV.setDepthValue(0)
            #    self.addItem(self.GLV, 'vol_total')

        else:
            if ev.buttons() == Qt.LeftButton:
                if (ev.modifiers() & Qt.ControlModifier):
                    self.pan(diff.x(), diff.y(), 0, relative='view')
                else:
                    self.orbit(-diff.x(), diff.y())
            elif ev.buttons() == Qt.MidButton:
                if (ev.modifiers() & Qt.ControlModifier):
                    self.pan(diff.x(), 0, diff.y(), relative='view-upright')
                else:
                    self.pan(diff.x(), diff.y(), 0, relative='view-upright')
            self.point3dpos.emit([self.opts['elevation'], self.opts['azimuth'], self.opts['distance']],None)
    def emit_viewport(self, points):
        self.point3dpos.emit(points, None)

    def mousePressEvent(self, ev):
        """
        By Bahram Jafrasteh
        :param ev:
        :return:
        """
        #self.opts['azimuth'] = 0
        #self.opts['elevation'] = 0

        self.mousePos = ev.localPos()
        if not (self._gotoLoc or self._enabledPolygon):
            return

        if ev.button()==Qt.RightButton:
            return
        if 'vol_total' in self.items:
            self.SubUpdateSegScatterItem()
        else:
            return
        points, _ = self._compute_coordinates(ev)

        #points[1] -= 1
        #print(points)

        if self._enabledPolygon:
            #self.addItem(self.GLPl, 'pol_total')
            #self.points = []
            #points[2] = self.maxXYZ[2] - points[2]
            #points[0] = self.maxXYZ[2] - points[0]
            p1, xo = self._compute_coordinates(ev, 1)
            self._sPoints = []
            self.totalpolys[self._numPolys] =[]
            if points[-2] == 0:
                return
            #print(points)
            x, y, z , _=points
            endIm = self._endIMage(x, y, z)
            if endIm[0]:
                # update x and y
                x = endIm[1]
                y = endIm[2]
                z = endIm[3]
            endIm2 = self._endIMage(p1[0], p1[2], p1[3])
            #if not endIm2[0]:
            #    self._sPoints.append(p1)

            self._sPoints.append(xo)
            self.totalpolys[self._numPolys].append([x, y, z])
            if len(self.totalpolys[self._numPolys])>1:
                self.GLPl.update_points(self.totalpolys)



        #print(points)
        elif self._gotoLoc:
            #points[0] -= 1
            #points[-1] -= 1
            points[2] = self.maxXYZ[0] - points[2]
            points[0] = self.maxXYZ[2] - points[0]
            if self.GLV._ax==0:
                windowName = 'sagittal'
            elif self.GLV._ax == 1:
                windowName = 'coronal'
            elif self.GLV._ax == 2:
                windowName = 'axial'
            self.point3dpos.emit(points, windowName)
        else:
            self.point3dpos.emit(points, None)
        #self.updateSegVolItem(self._seg_im, None)
        if 'scatter_total' in self.items:

            self.removeItem('scatter_total')
            self.GLV.setData(self._seg_im, self._artistic)
            self.GLV.setDepthValue(20)
            self.addItem(self.GLV, 'vol_total')



    """
    
    def changedata(self):
        self._artistic = False
        if self._UseScatter:
            self._UseScatter=False
            #self.paint(self._seg_im)
            self.SubUpdateSegScatterItem()
        elif not self._UseScatter:
            self._UseScatter=True

        self.update()
    """
    def _localUpdate(self):

        self.removeItem('scatter_total')
        if self._seg_im is not None:
            self.GLV.setData(self._seg_im.copy(), self._artistic)
        else:
            self.GLV.data = None
        self.GLV.setDepthValue(20)
        self.addItem(self.GLV, 'vol_total')

    def draw_art_action(self, selected_action_name):
        # Uncheck all other actions
        for action_name, action in self.action_objects.items():
            if action_name != selected_action_name:
                action.setChecked(False)
            else:
                action.setChecked(True)

        # Set the artistic value based on the selected action
        self._artistic = selected_action_name

        # Perform updates based on action states
        if self.seg_action.isChecked() or self.im_seg_action.isChecked():
            self.update_3dview.emit(True, None)
        elif hasattr(self, '_lastmap_type'):
            self.update_cmap_image(self._lastmap_type, True)
        self.update()

    def draw_status(self, value):
        try:
            if not value:
                self.removeItem('pol_total')
                self.totalpolys = defaultdict(list)
                totalpolys_camer = defaultdict(list)
                self.GLPl.update_points(self.totalpolys)
                self._enabledPolygon = value
            else:
                self.GLV.setDepthValue(0)
                self._enabledPolygon = value


                """
                
                if self.GLV._ax==0: #sagittal
                    if self.GLV._d==1:
                        self.opts['elevation'] = 0
                        self.opts['azimuth'] = 0
                    else:
                        self.opts['elevation'] = 0
                        self.opts['azimuth'] = -180
                elif self.GLV._ax==1:#coronal
                    if self.GLV._d == 1:
                        self.opts['elevation'] = 0
                        self.opts['azimuth'] = 90
                    else:
                        self.opts['elevation'] = 0
                        self.opts['azimuth'] = -90
                elif self.GLV._ax==2:#axial
                    if self.GLV._d == 1:
                        self.opts['elevation'] = 90
                        self.opts['azimuth'] = 0
                    else:
                        self.opts['elevation'] = -90
                        self.opts['azimuth'] = 0
                """
                self.SubUpdateSegScatterItem()
                self.showEvent(0)
                self._z = glReadPixels(0, 0, self.height(),self.width(), GL_DEPTH_COMPONENT, GL_FLOAT)
                if 'scatter_total' in self.items:
                    self.removeItem('scatter_total')
                    self.GLV.setData(self._seg_im, self._artistic)
                    self.GLV.setDepthValue(0)
                    self.addItem(self.GLV, 'vol_total')
        except Exception as e:
            print(e)
    def axis_status(self, value):
        if not value:
            self.removeItem('ax')
        else:
            self.addItem(self.ax,'ax')
            self.axis_action.setChecked(True)

    def grid_status(self, value):
        if not value:
            self.removeItem('xyz')
        else:
            self.addItem(self.gx, 'xyz')
            self.grid_action.setChecked(True)
    def artistic_status(self, value):
        self._artistic = True
        if not value:
            self._artistic = False
        self._localUpdate()


    def im_seg_status(self, value):
        self.seg_action.setChecked(False)
        if value:
            self._renderMode = 'Seg'
            self._threshold=0
            self.im_seg_action.setChecked(True)
            self.update_3dview.emit(True, None)


            #self.removeItem('vol_total')

        else:
            self._renderMode = 'Img'
            self.im_seg_action.setChecked(False)

            self.removeItem('vol_total')
            self.paintGL()



    def seg_status(self, value):
        self.im_seg_action.setChecked(False)

        if value:
            self._renderMode = 'Seg'
            self._threshold=0

            self.seg_action.setChecked(True)
            self.update_3dview.emit(True, None)

        else:
            self._renderMode = 'Img'
            self.seg_action.setChecked(False)
            self.removeItem('vol_total')
            self.paintGL()
        #if hasattr(self, 'GLV'):
        #    self.GLV.smooth = value
        #    self.GLV._needUpload = True
        #    self._localUpdate()
        #    self.paintGL()


    def goto_status(self, value):
        self._gotoLoc = value
        if not value:
            self.point3dpos.emit(np.array([0,0,0,1]),None)


    def changeBG(self, cl):
        self.opts['bgcolor'] = cl
        self._localUpdate()
        self.paintGL()


    def update_cmap_image(self, map_type, reset=True):
        self.update_cmap.emit(map_type, reset)

    def cmap_image(self, _image, map_type, reset=True):
        #if self._image is None:
        #    return
        if reset:
            self._threshold = 0
        self._renderMode = 'Seg'
        self.seg_action.setChecked(False)
        self.im_seg_action.setChecked(False)
        self._verticalSlider_1.setVisible(True)
        mask = _image<=(_image.max()*self._threshold/100)
        if map_type=='original':
            cm = np.repeat(_image[...,None], 4, -1)
        else:
            import matplotlib.pyplot as plt
            cm = plt.get_cmap(map_type, lut=256)
            cm = cm(_image / _image.max()) * 255.0
            #if map_type!='gist_rainbow_r':
            #    cm[...,3]=self._image
        self.removeItem('vol_total')

        self._seg_im = cm
        self._seg_im[mask,:]=0
        #if self._indices is not None:
        self._indices = (1-mask.astype('int'))>0
        self._indices = (self._indices.astype('int') - self._excluded_inds.astype('int')) > 0
        self._seg_im[self._excluded_inds,:]=0
        self.GLV.setData(self._seg_im, self._artistic)
        self.GLV.setDepthValue(20)
        self.addItem(self.GLV, 'vol_total')
        self.paintGL()
        self._lastmap_type = map_type
        self._rendered = 'image'

    def ShowContextMenu(self, pos):
        # Main Edit Menu
        menu = QMenu("Edit")

        # Submenu for Background Colors
        self.bgColorMenu = QMenu('Background Color')

        grc = QAction('Gray')
        gc = QAction("Green")
        oc = QAction("Orange")
        pc = QAction("Pink")
        vc = QAction("Violet")
        rc = QAction("Red")
        bc = QAction("Blue")
        wc = QAction("White")
        yc = QAction("Yellow")
        blc = QAction("Black")

        gc.triggered.connect(partial(self.changeBG, [0.5, 1, 0.0, 1]))
        oc.triggered.connect(partial(self.changeBG, [1, 0.7, 0.1, 1]))
        pc.triggered.connect(partial(self.changeBG, [1, 0.41, 0.79, 1]))

        vc.triggered.connect(partial(self.changeBG, [0.93, 0.51, 0.93, 1]))
        wc.triggered.connect(partial(self.changeBG, [1., 1., 1., 1]))#[0.96, 0.96, 0.86, 1]
        rc.triggered.connect(partial(self.changeBG, [0.98, 0.5, 0.44, 1]))
        bc.triggered.connect(partial(self.changeBG, [0.25, 0.87, 0.82, 1]))
        yc.triggered.connect(partial(self.changeBG, [1, 0.87, 0.0, 1]))
        blc.triggered.connect(partial(self.changeBG, [0.05, 0.05, 0.05, 1]))
        grc.triggered.connect(partial(self.changeBG, [0.3, 0.3, 0.3, 1]))
        self.bgColorMenu.addAction(gc)
        self.bgColorMenu.addAction(grc)
        self.bgColorMenu.addAction(rc)
        self.bgColorMenu.addAction(bc)
        self.bgColorMenu.addAction(vc)
        self.bgColorMenu.addAction(wc)
        self.bgColorMenu.addAction(oc)
        self.bgColorMenu.addAction(pc)
        self.bgColorMenu.addAction(yc)
        self.bgColorMenu.addAction(blc)


        # Submenu for Image Rendering/Colormap Options
        self.imageRenderMenu = QMenu("Image Render")

        gis_rainbow = QAction("RainBow")
        gis_rainbow.triggered.connect(partial(self.update_cmap_image, 'gist_rainbow_r'))

        gray = QAction("Gray")
        gray.triggered.connect(partial(self.update_cmap_image, 'gray_r'))

        original = QAction("Original")
        original.triggered.connect(partial(self.update_cmap_image, 'original'))

        jet = QAction("JET")
        jet.triggered.connect(partial(self.update_cmap_image, 'jet_r'))

        CMRmap = QAction("gnuplot")
        CMRmap.triggered.connect(partial(self.update_cmap_image, 'gnuplot2'))

        gnuplot_r = QAction("gnuplot2")
        gnuplot_r.triggered.connect(partial(self.update_cmap_image, 'gnuplot_r'))
        self.imageRenderMenu.addAction(gis_rainbow)
        self.imageRenderMenu.addAction(gray)
        self.imageRenderMenu.addAction(jet)
        self.imageRenderMenu.addAction(CMRmap)
        self.imageRenderMenu.addAction(gnuplot_r)
        self.imageRenderMenu.addAction(original)


        # Submenu for dividing the image
        self.CutMenu = QMenu('Cut')

        # Define the action names and labels in a list
        self.actions = [
            ('cut_remove_half_action', 'Remove Right Half'),
            ('cut_remove_left_half_action', 'Remove Left Half'),
            ('cut_remove_top_half_action', 'Remove Top Half'),
            ('cut_remove_bottom_half_action', 'Remove Bottom Half'),
            ('cut_remove_front_half_action', 'Remove Front Half'),
            ('cut_remove_back_half_action', 'Remove Back Half'),
            ('cut_remove_quarter_action', 'Remove Top-Right Quarter'),
            ('cut_remove_eighth_action', 'Remove Top-Right Front Eighth')
        ]

        # Create actions dynamically
        self.action_objects = {}
        for action_name, label in self.actions:
            action = QAction(label)
            action.setCheckable(True)
            action.setChecked(False)
            action.triggered.connect(partial(self.draw_art_action, action_name))
            self.CutMenu.addAction(action)
            self.action_objects[action_name] = action




        # Submenu for Segmentation Options
        self.segmentationMenu = QMenu('Segmentation')
        self.segmentationMenu.addAction(self.im_seg_action)
        self.segmentationMenu.addAction(self.seg_action)

        # Submenu for Drawing Options
        self.paintingMenu = QMenu('Painting')
        self.paintingMenu.addAction(self.draw_action)
        #self.paintingMenu.addAction(self.draw_art)
        self.paintingMenu.addAction(self.clear_action)
        self.paintingMenu.addMenu(self.imageRenderMenu)  # Nested menu for image rendering

        # View Options (Axis, Grid, Slice)
        self.viewMenu = QMenu("View Options")
        self.viewMenu.addAction(self.axis_action)
        self.viewMenu.addAction(self.grid_action)


        # Actions directly in the main menu
        menu.addMenu(self.bgColorMenu)  # Background Color
        menu.addMenu(self.paintingMenu)  # Painting and Image Rendering
        menu.addMenu(self.segmentationMenu)  # Segmentation
        menu.addMenu(self.viewMenu)  # View Options
        menu.addMenu(self.CutMenu)
        menu.addAction(self.screenshot_action)  # Screenshot
        menu.addAction(self.goto_action)  # GoTo


        menu.exec_(self.mapToGlobal(pos))



    def changeAxis(self, polygon, axis):
        if axis == 'coronal':
            polygon = polygon[:, [0, 2, 1]]
            # polygon = np.flipud(polygon)
            polygon[:, 0] = self.maxXYZ[2] - polygon[:, 0]
            polygon[:, 2] = self.maxXYZ[0] - polygon[:, 2]
        elif axis == 'sagittal':
            polygon = polygon[:, [2, 0, 1]]
            #polygon[:, 1] = self.maxXYZ[2] - polygon[:, 1]
            polygon[:, 2] = self.maxXYZ[0] - polygon[:, 2]
            #polygon[:,0] = -polygon[:,0]
            #polygon[:,2] = -polygon[:,2]
        elif axis == 'axial':
            polygon = polygon[:, [0, 1, 2]]
            polygon[:, 0] = self.maxXYZ[2] - polygon[:, 0]
            #polygon[:, 2] = self.maxXYZ[2] - polygon[:, 2]
        return polygon


    def updateSegSlice(self, imSeg, edges, currentWidnowName, sliceNum):
        """
        :param totalPs: keys of total points
        :param changedIndice: indices of key name
        :return:
        """
        minimum_area = 2
        if currentWidnowName.lower() == 'coronal':
            subim = imSeg[:,sliceNum,:]
            d = np.where(subim > 0)
            contours, hierarchy = cv2.findContours(image=subim.astype(np.uint8), mode=cv2.RETR_TREE,
                                                   method=cv2.CHAIN_APPROX_NONE)
            if len(contours)==0:
                item = GLScatterPlotItem(pos=np.empty(shape=(0,3), dtype=np.int64), color=np.empty(shape=(0,4), dtype=np.float64), pxMode=True, size=5)
                self.addItem(item, 'scatter_{}_{}'.format(currentWidnowName, sliceNum))
                return
            else:
                cnts = []
                i = 0
                for contour in contours:
                    if cv2.contourArea(contour)<minimum_area:
                        continue
                    cnt = contour.squeeze()
                    val = subim[cnt[0, 1], cnt[0, 0]]
                    if val > 150:
                        val -= 150
                        subc = subim[cnt[:, 1], cnt[:, 0]]
                        if sum(subc>150)>0:
                            subc[subc>150] = subc[subc>150] - 150
                            subim[cnt[:, 1], cnt[:, 0]] = subc
                    cnts.append([cnt, val])
                    subim[cnt[:, 1], cnt[:, 0]] = 150+subim[cnt[0, 1], cnt[0, 0]]
                    print(np.unique(subim))
                    print(i)
                    i+=1
                    print('')
            #cnt = contours[0].squeeze()

            #subim[cnt[:, 1], cnt[:, 0]] = 150+subim[cnt[0, 1], cnt[0, 0]]
            points = np.vstack((d[0], np.repeat(sliceNum,len(d[0])), d[1])).transpose([1,0])[:,[2,1,0]]
            #cnts = np.vstack((cnt[:, 1], np.repeat(sliceNum, cnt.shape[0]), cnt[:, 0] )).transpose([1, 0])[:, [2, 1, 0]]
        elif currentWidnowName.lower() == 'sagittal':
            subim = imSeg[:,:,sliceNum]
            d = np.where(subim > 0)
            contours, hierarchy = cv2.findContours(image=subim.astype(np.uint8), mode=cv2.RETR_TREE,
                                                   method=cv2.CHAIN_APPROX_NONE)
            if len(contours)==0:
                item = GLScatterPlotItem(pos=np.empty(shape=(0,3), dtype=np.int64), color=np.empty(shape=(0,4), dtype=np.float64), pxMode=True, size=5)
                self.addItem(item, 'scatter_{}_{}'.format(currentWidnowName, sliceNum))
                return
            else:
                cnts = []
                for contour in contours:
                    if cv2.contourArea(contour)<minimum_area:
                        continue
                    cnt = contour.squeeze()
                    val = subim[cnt[0, 1], cnt[0, 0]]
                    if val>150:
                        val -= 150
                        d = subim[cnt[:, 1], cnt[:, 0]]
                        if sum(d>150)>0:
                            d[d>150] = d[d>150] - 150
                            subim[cnt[:, 1], cnt[:, 0]] = d
                    cnts.append([cnt, val] )
                    subim[cnt[:, 1], cnt[:, 0]] = 150+subim[cnt[0, 1], cnt[0, 0]]
            points = np.vstack((d[0], d[1],np.repeat(sliceNum,len(d[0])))).transpose([1,0])[:,[2,1,0]]
            #cnts = np.vstack((cnt[:, 1], cnt[:, 0], np.repeat(sliceNum, cnt.shape[0]))).transpose([1, 0])[:, [2, 1, 0]]
        elif currentWidnowName.lower() == 'axial':
            subim = imSeg[sliceNum,:,:]
            d = np.where(subim > 0)
            contours, hierarchy = cv2.findContours(image=subim.astype(np.uint8), mode=cv2.RETR_TREE,
                                                   method=cv2.CHAIN_APPROX_NONE)
            if len(contours)==0:
                item = GLScatterPlotItem(pos=np.empty(shape=(0,3), dtype=np.int64), color=np.empty(shape=(0,4), dtype=np.float64), pxMode=True, size=5)
                self.addItem(item, 'scatter_{}_{}'.format(currentWidnowName, sliceNum))
                return
            else:
                cnts = []
                for contour in contours:
                    if cv2.contourArea(contour)<minimum_area:
                        continue
                    cnt = contour.squeeze()
                    val = subim[cnt[0, 1], cnt[0, 0]]
                    if val > 150:
                        val -= 150
                        d = subim[cnt[:, 1], cnt[:, 0]]
                        if sum(d>150)>0:
                            d[d>150] = d[d>150] - 150
                            subim[cnt[:, 1], cnt[:, 0]] = d
                    cnts.append([cnt, val])
                    subim[cnt[:, 1], cnt[:, 0]] = 150+subim[cnt[0, 1], cnt[0, 0]]
            #cnt = contours[0].squeeze()
            #subim[cnt[:, 1], cnt[:, 0]] = 150+subim[cnt[0, 1], cnt[0, 0]]
            points = np.vstack((np.repeat(sliceNum,len(d[0])),d[0], d[1])).transpose([1,0])[:,[2,1,0]]
            #cnts = np.vstack((np.repeat(sliceNum, cnt.shape[0]), cnt[:, 1], cnt[:, 0] )).transpose([1, 0])[:, [2, 1, 0]]


        #d = np.where(imSeg>0)
        #if d[0].shape[0]<=1:
            #return
        #points = np.vstack((d[0], d[1], d[2])).transpose([1, 0])[:,[2,1,0]]

        points[:, 0] = self.maxXYZ[2] - points[:, 0]
        points[:, 2] = self.maxXYZ[0] - points[:, 2]

        #cnts[:, 0] = self.maxXYZ[2] - cnts[:, 0]
        #cnts[:, 2] = self.maxXYZ[0] - cnts[:, 2]


        """
        if windowName == 'coronal':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'sagittal':
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'axial':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
        """
        colors = np.zeros((points.shape[0], 4))
        #colors[:, 3] = imSeg[tuple(zip(d))].squeeze()
        #colors[:, 2] = data[tuple(zip(d))].squeeze()
        #colors[:, 1] = data[tuple(zip(d))].squeeze()
        colorsInd =  subim[tuple(zip(d))].transpose().squeeze()
        for cl in np.unique(colorsInd[colorsInd<=150]):
            if cl == 0:
                continue
            ind = colorsInd == cl
            colorval = self.colorsCombinations[int(cl)]
            colors[ind,:] =colorval

            ind_edge = colorsInd == cl+150
            colorval_edge = (0.,0.,0.,1.0)
            colors[ind_edge, :] = colorval_edge
            for cnt0 in cnts:
                cnt, cl = cnt0
                subim[cnt[:, 1], cnt[:, 0]] = cl
        #colors[:,0] = 255.0
        #colors = colors/255.0
        item = GLScatterPlotItem(pos=points, color = colors, pxMode=True, size=5)
        self.addItem(item, 'scatter_{}_{}'.format(currentWidnowName, sliceNum))

        return


    def updateSegVolItem(self, imSeg = None, imOrg = None, currentWidnowName=None, sliceNum = None):
        """
        :param totalPs: keys of total points
        :param changedIndice: indices of key name
        :return:
        """
        self.removeItem('scatter_total')
        d = np.where(imSeg>0)
        if d[0].shape[0]<=1:
            return
        self.maxXYZ_current = [d[0].max(),d[1].max(),d[2].max()]

        self.minXYZ_current = [d[0].min(),d[1].min(),d[2].min()]
        if d[0].shape[0]<=1:
            self.removeItem('scatter_total')
            return


        """
        if windowName == 'coronal':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'sagittal':
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'axial':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
        """

        #colors[:, 3] = imSeg[tuple(zip(d))].squeeze()
        #colors[:, 2] = data[tuple(zip(d))].squeeze()
        #colors[:, 1] = data[tuple(zip(d))].squeeze()
        colorsInd =  imSeg[tuple(zip(d))].squeeze()
        uq = np.unique(colorsInd)
        if 9876 not in self.colorInds:  # len(self.colorsCombinations):
            selected_ud = self.colorInds
        else:
            selected_ud = uq
        _seg_im = np.zeros((*imSeg.shape, 4))
        indics = 0
        for cl in uq:
            if cl in selected_ud:
                ind = imSeg == cl
                indics += ind
                try:
                    colorval = self.colorsCombinations[int(cl)]
                    _seg_im[ind, :] = colorval
                except:
                    print('Index {} does not have a representative color.'.format(int(cl)))
        #colors[:,0] = 255.0
        #colors = colors/255.0
        #if self._indices is None:
        self._indices = indics>self._threshold
        self._rendered = 'seg'
        self._verticalSlider_1.setVisible(False)
        _seg_im *= 255
        _seg_im[self._excluded_inds,:]=0
        #im_or = np.repeat(imOrg[..., None], [4], axis=-1)
        #im_or[indics<=0,:]=0
        #a = self.intensitySeg * _seg_im + (1 - self.intensitySeg) * im_or

        if self.im_seg_action.isChecked():
            cm = np.repeat(imOrg[..., None], 4, -1)
            mask = imOrg <= (imOrg.max() * self._threshold / 100)

            cm_rgb = cm[...,:3]/255.0
            #cm_rgb[imSeg>0,:]=0 # remove segmented ROIs from the original image
            cm_alpha = cm[...,3]/255.0

            seg_rgb = _seg_im[...,:3]/255.0
            seg_alpha = _seg_im[..., 3]/ 255.0

            user_threshold = self.intensityImg
            #cm_alpha *= user_threshold if user_threshold else 0.02
            """
            

            # Calculate the gradient along each dimension (x, y, z)
            gradients = np.gradient(cm[..., :3], axis=(0, 1, 2))

            # Compute the gradient magnitude
            gradient_magnitude = np.sqrt(sum(g ** 2 for g in gradients)).mean(axis=-1)

            gradient_magnitude = gaussian_filter(gradient_magnitude, sigma=2)

            # Normalize the gradient magnitude to [0, 1]
            gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (
                    gradient_magnitude.max() - gradient_magnitude.min()
            )
            gradient_magnitude*=0.5
            # Adjust alpha based on the gradient magnitude
            #cm_alpha *= (gradient_magnitude ** 2)  # Emphasize significant gradients more
            """
            cm_alpha*=self.intensityImg

            #seg_alpha *= 1.3 # linear
            seg_alpha = np.sqrt(seg_alpha) # non linear

            composite_alpha = cm_alpha + seg_alpha *(1-cm_alpha)

            composite_alpha_clipped = np.clip(composite_alpha, a_min=1e-10, a_max=None)
            composite_rgb = (cm_rgb * cm_alpha[..., np.newaxis] + seg_rgb * seg_alpha[..., np.newaxis] * (
                        1 - cm_alpha[..., np.newaxis])) / composite_alpha_clipped[..., np.newaxis]
            composite_rgb[composite_alpha == 0] = 0
            composite_alpha[composite_alpha == 0] = 1
            _seg_im = np.concatenate((composite_rgb*255.0, composite_alpha[..., np.newaxis] * 255), axis=-1)
            _seg_im[mask,:] = 0
            #self._seg_im = _seg_im
            #self.GLV.setData(_seg_im, self._artistic)

        self._seg_im = _seg_im
        self.GLV.setData(_seg_im, self._artistic)

        self.GLV.setDepthValue(20)
        self.addItem(self.GLV, 'vol_total')

        return



    def SubUpdateSegScatterItem(self):
        if self._seg_im is None:
            return
        if self._indices is None:
            self._indices = self._image>0
        self.removeItem('vol_total')
        d = np.where(self._indices)
        if d[0].shape[0]<=1:
            return
        points = np.vstack((d[0], d[1], d[2])).transpose([1, 0])  # [:,[2,1,0]]

        points[:, 0] = self.maxXYZ[0] - points[:, 0]  # axial
        points[:, 2] = self.maxXYZ[2] - points[:, 2]  # sagittal
        # points[:, 1] = self.maxXYZ[1] - points[:, 1]#coronal
        points = points[:, [2, 1, 0]]
        colors = self._seg_im[self._indices]/255.0

        self.GLSC.setData(pos=points, color = colors, pxMode=True, size=5)
        self.addItem(self.GLSC, 'scatter_total')

    def updateSegScatterItem(self, imSeg, windowName):
        """
        :param totalPs: keys of total points
        :param changedIndice: indices of key name
        :return:
        """
        self.removeItem('vol_total')

        d = np.where(imSeg>0)
        if d[0].shape[0]<=1:
            self.removeItem('scatter_total')
            return
        points = np.vstack((d[0], d[1], d[2])).transpose([1, 0])#[:,[2,1,0]]

        points[:, 0] = self.maxXYZ[0] - points[:, 0]#axial
        points[:, 2] = self.maxXYZ[2] - points[:, 2]#sagittal
        #points[:, 1] = self.maxXYZ[1] - points[:, 1]#coronal
        points = points[:,[2,1,0]]

        """
        if windowName == 'coronal':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'sagittal':
            points[:, 2] = self.maxXYZ[0] - points[:, 2]
        elif windowName == 'axial':
            points[:, 0] = self.maxXYZ[2] - points[:, 0]
        """
        colors = np.ones((points.shape[0], 4))*10000
        #colors[:, 3] = imSeg[tuple(zip(d))].squeeze()
        #colors[:, 2] = data[tuple(zip(d))].squeeze()
        #colors[:, 1] = data[tuple(zip(d))].squeeze()
        colorsInd =  imSeg[tuple(zip(d))].squeeze()
        uq = np.unique(colorsInd)
        if 9876 not in self.colorInds:  # len(self.colorsCombinations):
            selected_ud = self.colorInds
        else:
            selected_ud = uq

        for cl in uq:
            if cl == 0:
                continue
            if cl in selected_ud:
                ind = colorsInd == cl
                try:
                    colorval = self.colorsCombinations[int(cl)]
                    colors[ind,:] =colorval
                except:
                    print('Index {} does not have a representative color.'.format(int(cl)))
        #colors[:,0] = 255.0
        #colors = colors/255.0
        ind_non_zero = colors.sum(1)!=10000*4
        self.GLSC.setData(pos=points[ind_non_zero, :], color = colors[ind_non_zero, :], pxMode=True, size=5)
        self.addItem(self.GLSC, 'scatter_total')
        return
        if len(changedIndice)== 0:
            return

        for axis in totalPs.keys():
            for Pls in totalPs[axis].keys():
                for key in totalPs[axis][Pls].keys():
                    keyName = str(axis) + '_' + str(Pls) + '_' + str(key)
                    if keyName in changedIndice:
                        ind = changedIndice.index(keyName)
                        self.removeItem(keyName)
                        polyg, color = totalPs[axis][Pls][key]
                        polygon = np.array(list(polyg.exterior.coords))
                        polygon = self.changeAxis(polygon, axis)
                        keyName = str(axis) + '_' + str(Pls) + '_' + str(key)
                        polygon = np.array(PolygonTessellator().tessellate(polygon))
                        self.addSegItem(polygon, keyName, color)
                        changedIndice.pop(ind)
        if len(changedIndice) != 0:
            for keyName in changedIndice:
                self.removeItem(keyName)




    #def paint(self, imSeg = None, edges = None, currentWidnowName=None, sliceNum = None):
    def paint(self, imSeg = None, im = None, currentWidnowName=None, sliceNum = None):
        if not self._updatePaint:
            return
        if not self._renderMode.lower()=='seg':
            return
        if not (self.seg_action.isChecked() or self.im_seg_action.isChecked()):
            return
        #self._seg_im = imSeg
        #if imSeg is not None:
            #if currentWidnowName is None:
        #if self._UseScatter:
        #    self.updateSegScatterItem(imSeg, currentWidnowName)
        #else:

        self.updateSegVolItem(imSeg, im, currentWidnowName)
            #else:
            #    self.updateSegSlice(imSeg, edges, currentWidnowName, sliceNum)

        if self.isVisible():
            self.show()


    def load_paint(self, imSeg):

        if hasattr(self,'items_names'):
            if len(self.items_names)>3:
                try:
                    total_items = list(set(self.items_names) - set(['scatter_total', 'xyz', 'ax']))
                    for item in total_items:
                        _, currentWidnowName, sliceNum = item.split('_')
                        sliceNum = int(sliceNum)
                        self.updateSegSlice(imSeg, [], currentWidnowName, sliceNum)
                    delattr(self, 'items_names')
                    self.update()
                except:
                    self.paint(imSeg, None)
            else:
                self.paint(imSeg, None)
        if hasattr(self, 'items_names'):
            delattr(self, 'items_names')


    def createGridAxis(self, maxcoord):
        self.removeItem('vol_total')
        #maxcoord = self.testTest()
        ############# create the background grids #############
        self.setMaxCoords(maxcoord)

        #self._seg_im = np.zeros((*maxcoord,1)).squeeze()
        if len(self.totalItems['axis']) != 0:
            self.removeItem(self.totalItems['axis'][0])
            self.removeItem(self.totalItems['axis'][1])
            #self.removeItem(self.totalItems['axis'][2])
            #self.removeItem(self.totalItems['axis'][3])
            del self.totalItems['axis']
        zmax, ymax, xmax = maxcoord
        self.maxXYZ = [zmax, ymax, xmax]
        self._excluded_inds = np.zeros((self.maxXYZ[0],self.maxXYZ[1],self.maxXYZ[2]), bool)
        self._excluded_inds[:]=False
        xvals = np.arange(0, xmax + xmax/10, xmax/5).astype(np.int64)
        yvals = np.arange(0, ymax + ymax/10, ymax/5).astype(np.int64)
        zvals = np.arange(0, zmax + zmax/10, zmax/5).astype(np.int64)

        self.maxXYZ = [zmax, ymax, xmax]
        self.maxXYZ_current = [zmax, ymax, xmax]
        self.minXYZ_current = [0, 0, 0]
        #xvals = -np.arange(0, xmax + xmax/5, xmax/5).astype("int")[::-1]
        #yvals = -np.arange(0, ymax + ymax/5, ymax/5).astype("int")[::-1]
        #zvals = -np.arange(0, zmax + zmax/5, zmax/5).astype("int")[::-1]

        zmax = zvals[-1]; ymax = yvals[-1]; xmax=xvals[-1]
        zmin = zvals[0];ymin = yvals[0];xmin = xvals[0]
        self.gx = GLGridItem(glOptions='translucent', color=[0,0,0,1])
        self.gx.setGrid(xvals, yvals, zvals )
        self.gx.setDepthValue(0)





        #gx.translate(-zmax, 0,  0)
        #gx.rotate(90, 0, 1, 0)
        #gx.translate(0, 0, zmax)
        if self.grid_action.isChecked():
            self.totalItems['axis'].append('xyz')
            self.addItem(self.gx, 'xyz')

        #imxz = np.zeros((xmax, zmax,3))
        #Imxz = GLImageItem(imxz, color=[1,1,0,1], program=self.program)
        #Imxz.rotate(90,0,1,0)
        #self.addItem(Imxz, 'imxz')

        #gy = gl.GLGridItem(glOptions='opaque', program=self.program)
        #gy.rotate(90, 1, 0, 0)
        #gy.setGrid(xvals, zvals)
        #self.totalItems['axis'].append('y')
        #self.addItem(gy, 'y')

        #gz = gl.GLGridItem(glOptions='opaque', program=self.program)
        #gz.setGrid(xvals, yvals)
        #self.totalItems['axis'].append('z')
        #self.addItem(gz, 'z')

        ############# axis #############
        self.ax = GLAxisItem(antialias=True, glOptions='translucent')
        self.ax.setDepthValue(0)
        self.ax.setGLViewWidget(self)
        #ax.setSize(0, xmax, 0, ymax, 0, zmax)
        self.ax.setSize(xmin, xmax, ymin,ymax, zmin, zmax)
        self.ax.setOriging(maxcoord[2]/2,maxcoord[1]/2,maxcoord[0]/2)
        if self.axis_action.isChecked():
            self.totalItems['axis'].append('ax')
            self.addItem(self.ax, 'ax')

        # Switch to 'nearly' orthographic projection.
        #self.opts['distance'] = 40000
        self.opts['fov'] = 45
        self.opts['elevation'] = 45
        self.opts['azimuth'] = 45

        self.setCameraPosition(pos=QVector3D(maxcoord[2]/2,maxcoord[1]/2,maxcoord[0]/2), rotation=(0,0,1,1),
                               distance= max(xmax, ymax, zmax))
                               #distance=np.sqrt(maxcoord[0]**2+maxcoord[1]**2+maxcoord[2]**2),
                               #)
        self.totalPolyItems = []

        self._enabledPolygon = False
        self._indices = None
        self._rendered = 'seg'
        self._numPolys = 0
        self.totalpolys = defaultdict(list)
        self.totalpolys_camer = defaultdict(list)






# Start Qt event loop unless running in interactive mode.
class Ui_Main0():
    NumRows = 2
    NumColumns = 3

    def __init__(self):
        super(Ui_Main0, self).__init__()

        self.glWidgets = []

    def setupUi(self, Main):
        Main.resize(2000, 1800)
        self.centralwidget = QWidget(Main)
        self.centralwidget.setEnabled(True)
        self.v = glScientific(self.centralwidget)


        import pickle
        #with open('dictionary', 'wb') as output:
        #    pickle.dump(self.colorsCombinations, output, pickle.HIGHEST_PROTOCOL)
        with open('dictionary', 'rb') as output:
            r = pickle.load(output)

        self.v.colorsCombinations = r
        self.v.colorInds = [85,45]
        #v.animation()

class MainWindow0(QWidget, Ui_Main0):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow0, self).__init__(*args, **kwargs)
        #QtGui.QWidget.__init__(self)
        self.setupUi(self)
        from nibabel import load
        #nii = load('inp.nii')
        #data = np.squeeze(nii.get_data())

        data = np.load('aa.npy')
        data = data.astype("float")  # Typecast to float
        #data = data[:, :, ::-1]
        self.v._renderMode = 'seg'
        self.v._excluded_inds = np.zeros_like(data).astype('bool')
        self.v._excluded_inds[:] = False
        self.v.createGridAxis(list(data.shape))
        self.v.paint(data)
        self.v.show()

        data = np.rot90(data, axes=(1, 2))# x
        data = np.rot90(data, axes=(0, 2)) #y
        data = np.rot90(data, axes=(0, 1))  # z

        #data = np.load('ab.npy')
        #data = data.astype("float")  # Typecast to float





if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow0()

    #window.show()
    sys.exit(app.exec_())



