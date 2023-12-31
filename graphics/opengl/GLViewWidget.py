
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
import numpy as np
from OpenGL.GLUT import *
from collections import defaultdict
from . import functions as fn
Vector = QtGui.QVector3D
ShareWidget = None

__AUTHOR__ = 'Bahram Jafrasteh'
"""
Inspiration from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""


class GLViewWidget(QOpenGLWidget):
    """
    Basic widget for displaying 3D data
        - Rotation/scale controls
        - Axis/grid display
        - Export options

    High-DPI displays: Qt5 should automatically detect the correct resolution.
    For Qt4, specify the ``devicePixelRatio`` argument when initializing the
    widget (usually this value is 1-2).
    """
    
    def __init__(self, parent=None, devicePixelRatio=None):
        global ShareWidget

        if ShareWidget is None:
            ## create a dummy widget to allow sharing objects (textures, shaders, etc) between views
            ShareWidget = QOpenGLWidget()
            

        QtWidgets.QOpenGLWidget.__init__(self, parent)
        
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.opts = {
            'devicePixelRatio': devicePixelRatio
        }
        self.reset()
        self.items = defaultdict(list)
        
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)
        
        self.makeCurrent()
        self.installEventFilter(self)
        
    def reset(self):
        """
        Initialize the widget state or reset the current state to the original state.
        """
        self.opts['center'] = Vector(0,0,0)  ## will always appear at the center of the widget
        self.opts['distance'] = 10.0         ## distance of camera from center
        self.opts['fov'] = 60                ## horizontal field of view in degrees
        self.opts['elevation'] = 30          ## camera's angle of elevation in degrees
        self.opts['azimuth'] = 45            ## camera's azimuthal angle in degrees 
                                             ## (rotation around z-axis 0 points along x-axis)
        self.opts['viewport'] = None         ## glViewport params; None == whole widget


    def addItem(self, item, itemKey):
        """
        By Bahram Jafrasteh
        :param item:
        :param itemKey:
        :return:
        """
        if itemKey in self.items:
            self.removeItem(itemKey)
        self.items[itemKey] = item

        if hasattr(item, 'initializeGL'):
            self.makeCurrent()
            try:
                item.initializeGL()
            except:
                self.checkOpenGLVersion('Error while adding item %s to GLViewWidget.' % str(item))

        item._setView(self)
        #print "set view", item, self, item.view()
        self.update()




    def removeItem(self, itemKey):
        """
        By Bahram Jafrasteh
        Remove the item from the scene.
        """
        #self.items.remove(item)
        if itemKey not in self.items:
            return
        self.items[itemKey]._setView(None)
        self.items.pop(itemKey)
        #item._setView(None)
        self.update()


    def clear(self):
        """
        Remove all items from the scene.
        """
        keys = []
        for key in self.items.keys():
            if key == 'xyz' or key == 'ax':
                continue
            item = self.items[key]
            item._setView(None)
            keys.append(key)
        for key in keys:
            self.items.pop(key)
        #self.items = defaultdict(list)
        self.update()
        
    def initializeGL(self):
        self.resizeGL(self.width(), self.height())
        

        
    def getViewport(self):
        vp = self.opts['viewport']
        dpr = self.devicePixelRatio()
        if vp is None:
            return (0, 0, int(self.width() * dpr), int(self.height() * dpr))
        else:
            return tuple([int(x * dpr) for x in vp])

    def _updateScreen(self, screen):
        """
        By Bahram Jafrasteh
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

    def showEvent(self, event):
        """
        By Bahram Jafrasteh
        :param event:
        :return:
        """
        window = self.window().windowHandle()
        window.screenChanged.connect(self._updateScreen)
        self._updateScreen(window.screen())

    def mousePressEvent(self, ev):
        """
        By Bahram Jafrasteh
        :param ev:
        :return:
        """
        self.mousePos = ev.localPos()

        self.showEvent(0)
        z = glReadPixels(ev.x(), self.height() - ev.y(), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]

        per = np.array(self.m.copyDataTo()).reshape((4, 4)).transpose().flatten()
        zFar = 0.5 * per[14] * (1.0 - ((per[10] - 1.0) / (per[10] + 1.0)))
        zNear = zFar * (per[10] + 1.0) / (per[10] - 1.0)

        ndcx = 2 * self.mousePos.x() / self.width() - 1
        ndcy = 1 - 2 * self.mousePos.y() / self.height()
        points = np.matmul(np.linalg.inv(self.mvpProj.reshape((4, 4)).transpose()), np.array([ndcx,ndcy, 2*z-1, 1]))
        points /= points[3]

        #self.point3dpos.emit(points)
        print(points)

    def devicePixelRatio(self):
        dpr = self.opts['devicePixelRatio']
        if dpr is not None:
            return dpr
        
        if hasattr(QOpenGLWidget, 'devicePixelRatio'):
            return QOpenGLWidget.devicePixelRatio(self)
        else:
            return 1.0
        
    def resizeGL(self, w, h):
        pass
        #glViewport(*self.getViewport())
        #self.update()

    def setProjection(self, region=None):

        self.m = self.projectionMatrix(region)
        self.a = self.viewMatrix()

        ma = self.m * self.a
        self.mvpProj = np.array(ma.copyDataTo()).reshape((4, 4)).transpose().flatten()

        #m = self.projectionMatrix(region)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        a = np.array(self.m.copyDataTo()).reshape((4,4))
        glMultMatrixf(a.transpose())

    def projectionMatrix(self, region=None):
        if region is None:
            dpr = self.devicePixelRatio()
            region = (0, 0, self.width() * dpr, self.height() * dpr)
        
        x0, y0, w, h = self.getViewport()
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.001
        farClip = dist * 1000.

        r = nearClip * np.tan(fov * 0.5 * np.pi / 180.)
        t = r * h / w

        ## Note that X0 and width in these equations must be the values used in viewport
        left  = r * ((region[0]-x0) * (2.0/w) - 1)
        right = r * ((region[0]+region[2]-x0) * (2.0/w) - 1)
        bottom = t * ((region[1]-y0) * (2.0/h) - 1)
        top    = t * ((region[1]+region[3]-y0) * (2.0/h) - 1)

        tr = QtGui.QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr
        
    def setModelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        #self.a = self.viewMatrix()
        a = np.array(self.a.copyDataTo()).reshape((4,4))
        glMultMatrixf(a.transpose())
        
    def viewMatrix(self):
        tr = QtGui.QMatrix4x4()
        tr.translate( 0.0, 0.0, -self.opts['distance'])
        tr.rotate(self.opts['elevation']-90, 1, 0, 0)
        tr.rotate(self.opts['azimuth']+90, 0, 0, -1)
        center = self.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr


    def paintGL(self, region=None, viewport=None, useItemNames=False):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        if viewport is None:
            glViewport(*self.getViewport())
        else:
            glViewport(*viewport)
        self.setProjection(region=region)
        self.setModelview()
        bgcolor = self.opts['bgcolor']


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT)
        #glClearColor(bgcolor[0], bgcolor[1], bgcolor[2], 0.5)
        glClearColor(bgcolor[0], bgcolor[1], bgcolor[2], bgcolor[3])

        self.drawItemTree(useItemNames=useItemNames)
        
    def drawItemTree(self, item=None, useItemNames=False):
        if item is None:
            items = [self.items[key] for key in self.items.keys() if self.items[key].parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        #items1 = [item for item in items if
        # hasattr(item, 'sliceDensity')]
        #items2 =[item for item in items if not hasattr(item, 'sliceDensity')]
        items.sort(key=lambda a: a.depthValue())

        #items= items1 + items2

        for i in items:


            if not i.visible():
                continue
            if i is item:
                try:
                    glPushAttrib(GL_ALL_ATTRIB_BITS)
                    if useItemNames:
                        glLoadName(i._id)
                        self._itemNames[i._id] = i
                    i.paint()
                except Exception as e:
                    #from .. import debug
                    #debug.printExc()
                    print(e)
                    msg = "Error while drawing item %s." % str(item)
                    ver = glGetString(GL_VERSION)
                    if ver is not None:
                        ver = ver.split()[0]
                        if int(ver.split(b'.')[0]) < 2:
                            print(msg + " The original exception is printed above; however, pyqtgraph requires OpenGL version 2.0 or greater for many of its 3D features and your OpenGL version is %s. Installing updated display drivers may resolve this issue." % ver)
                        else:
                            print(msg)
                    
                finally:
                    glPopAttrib()
            else:
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                try:
                    tr = i.transform()
                    a = np.array(tr.copyDataTo()).reshape((4,4))
                    glMultMatrixf(a.transpose())
                    self.drawItemTree(i, useItemNames=useItemNames)
                finally:
                    glMatrixMode(GL_MODELVIEW)
                    glPopMatrix()
            
    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None, rotation=None):
        """
        By Bahram Jafrasteh
        :param pos:
        :param distance:
        :param elevation:
        :param azimuth:
        :param rotation:
        :return:
        """
        if pos is not None:
            self.opts['center'] = pos
        if distance is not None:
            self.opts['distance'] = distance
        if rotation is not None:
            # set with quaternion
            self.opts['rotation'] = rotation
        else:
            # set with elevation-azimuth, restored for compatibility
            eu = self.opts['rotation'].toEulerAngles()
            if azimuth is not None:
                eu.setZ(-azimuth-90)
            if elevation is not None:
                eu.setX(elevation-90)
            self.opts['rotation'] = QtGui.QQuaternion.fromEulerAngles(eu)
        self.update()
        
    def cameraPosition(self):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        center = self.opts['center']
        dist = self.opts['distance']
        elev = self.opts['elevation'] * np.pi/180.
        azim = self.opts['azimuth'] * np.pi/180.
        
        pos = Vector(
            center.x() + dist * np.cos(elev) * np.cos(azim),
            center.y() + dist * np.cos(elev) * np.sin(azim),
            center.z() + dist * np.sin(elev)
        )
        
        return pos

    def orbit(self, azim, elev):
        """Orbits the camera around the center position. *azim* and *elev* are given in degrees."""
        self.opts['azimuth'] += azim
        self.opts['elevation'] = np.clip(self.opts['elevation'] + elev, -90, 90)
        self.update()
        
    def pan(self, dx, dy, dz, relative='global'):
        """
        Moves the center (look-at) position while holding the camera in place. 
        
        ==============  =======================================================
        **Arguments:**
        *dx*            Distance to pan in x direction
        *dy*            Distance to pan in y direction
        *dz*            Distance to pan in z direction
        *relative*      String that determines the direction of dx,dy,dz. 
                        If "global", then the global coordinate system is used.
                        If "view", then the z axis is aligned with the view
                        direction, and x and y axes are inthe plane of the
                        view: +x points right, +y points up. 
                        If "view-upright", then x is in the global xy plane and
                        points to the right side of the view, y is in the
                        global xy plane and orthogonal to x, and z points in
                        the global z direction.
        ==============  =======================================================
        
        Distances are scaled roughly such that a value of 1.0 moves
        by one pixel on screen.
        
        Prior to version 0.11, *relative* was expected to be either True (x-aligned) or
        False (global). These values are deprecated but still recognized.
        """
        # for backward compatibility:
        relative = {True: "view-upright", False: "global"}.get(relative, relative)
        
        if relative == 'global':
            self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
        elif relative == 'view-upright':
            cPos = self.cameraPosition()
            cVec = self.opts['center'] - cPos
            dist = cVec.length()  ## distance from camera to center
            xDist = dist * 2. * np.tan(0.5 * self.opts['fov'] * np.pi / 180.)  ## approx. width of view at distance of center point
            xScale = xDist / self.width()
            zVec = QtGui.QVector3D(0,0,1)
            xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
            yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
            self.opts['center'] = self.opts['center'] + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
        elif relative == 'view':
            # pan in plane of camera
            elev = np.radians(self.opts['elevation'])
            azim = np.radians(self.opts['azimuth'])
            fov = np.radians(self.opts['fov'])
            dist = (self.opts['center'] - self.cameraPosition()).length()
            fov_factor = np.tan(fov / 2) * 2
            scale_factor = dist * fov_factor / self.width()
            z = scale_factor * np.cos(elev) * dy
            x = scale_factor * (np.sin(azim) * dx - np.sin(elev) * np.cos(azim) * dy)
            y = scale_factor * (np.cos(azim) * dx + np.sin(elev) * np.sin(azim) * dy)
            self.opts['center'] += QtGui.QVector3D(x, -y, z)
        else:
            raise ValueError("relative argument must be global, view, or view-upright")
        
        self.update()
        
    def pixelSize(self, pos):
        """
        Return the approximate size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        cam = self.cameraPosition()
        if isinstance(pos, np.ndarray):
            cam = np.array(cam).reshape((1,)*(pos.ndim-1)+(3,))
            dist = ((pos-cam)**2).sum(axis=-1)**0.5
        else:
            dist = (pos-cam).length()
        xDist = dist * 2. * np.tan(0.5 * self.opts['fov'] * np.pi / 180.)
        return xDist / self.width()
        

        
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        
        if ev.buttons() == QtCore.Qt.LeftButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MidButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        


        
    def wheelEvent(self, ev):
        delta = 0

        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self.opts['fov'] *= 0.999**delta
        else:
            self.opts['distance'] *= 0.999**delta
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
      
    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        
    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key_Down:
                    self.orbit(azim=0, elev=speed)
                elif key == QtCore.Qt.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def checkOpenGLVersion(self, msg):
        """
        Give exception additional context about version support.

        Only to be called from within exception handler.
        As this check is only performed on error,
        unsupported versions might still work!
        """

        # Check for unsupported version
        verString = glGetString(GL_VERSION)
        ver = verString.split()[0]
        # If not OpenGL ES...
        if str(ver.split(b'.')[0]).isdigit():
            verNumber = int(ver.split(b'.')[0])
            # ...and version is supported:
            if verNumber >= 2:
                # OpenGL version is fine, raise the original exception
                raise



        # Notify about unsupported version
        raise Exception(
            msg + "\n" + \
            "pyqtgraph.opengl: Requires >= OpenGL 2.0 (not ES); Found %s" % verString
        )



    def setMaxCoords(self, maxCoord):
        self.maxCoord = maxCoord


    def renderText(self, text='', pos=(0,0,0), size=(1,1), rotation=(0,0,0)):
        """
        By Bahram Jafrasteh
        :param text: text to be printed
        :param pos: 3d position of the text
        :param size: width and height of the text
        :param rotation: 3d rotation angles
        :return: 3d text
        """


        glutInit()
        glPushMatrix()


        width, height = size
        #g = 0
        glRotatef(rotation[0], 1, 0, 0)
        glRotatef(rotation[1], 0, 1, 0)
        glRotatef(rotation[2], 0, 0, 1)
        glRasterPos3f(pos[0], pos[1], pos[2])
        l = list(text)



        for char in l:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, ord(char))

            glTranslatef(width, 0, 0)
            #glScale(10,10,10)

        glPopMatrix()
        return




