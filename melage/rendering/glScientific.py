# -*- coding: utf-8 -*-
__author__= 'Bahram Jafrasteh'
import numpy as np
import sys
sys.path.append('../')
from melage.rendering import GLViewWidget, GLAxisItem, GLScatterPlotItem, GLGridItem, GLVolumeItem, GLPolygonItem
from OpenGL.GL import *
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QAction, QSlider, QFileDialog, QLabel
from PyQt5.QtGui import QVector3D, QMatrix4x4, QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QRect
import cv2
from PyQt5.QtWidgets import QToolBar, QToolButton, QVBoxLayout, QFrame, QHBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
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
        self.id = id
        self._lastZ = 0.5  # Default to middle of the scene
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



        self._seg_im = None

        #self.opts['bgcolor'] = [0.5, 1, 0.0, 1]
        self.opts['bgcolor'] = [0.05, 0.05, 0.05, 1]#[0.3, 0.3, 0.3, 1] # background color

        self._verticalSlider_1 = QSlider(self)
        self._verticalSlider_1.setOrientation(Qt.Vertical)
        self._verticalSlider_1.setObjectName("verticalSlider_6")
        self._verticalSlider_1.setRange(1, 50)
        self._verticalSlider_1.setValue(1)
        self._verticalSlider_1.setVisible(False)
        #self.label_1 = QLabel(self)
        #self.label_1.setAlignment(Qt.AlignCenter)
        #self.label_1.setObjectName("label_1")



        self.initiate_actions()  # Make sure actions exist first
        self.init_overlay_toolbar()  # <--- Add this line here

        self._verticalSlider_1.valueChanged.connect(self.lbl_thresh.setNum)
        self._verticalSlider_1.valueChanged.connect(self.threshold_change)





    def init_overlay_toolbar(self):
        """
        Creates a transparent overlay toolbar in the top-left corner.
        Scales proportionally to the window size.
        """
        # --- 0. Initialize Timer for Smooth Orbiting ---
        if not hasattr(self, 'orbit_timer'):
            self.orbit_timer = QTimer(self)
            self.orbit_timer.timeout.connect(self._process_orbit_tick)

        # --- 1. Create Container ---
        if hasattr(self, 'overlay_frame'):
            self.overlay_frame.close()
            self.overlay_frame.deleteLater()

        self.overlay_frame = QFrame(self)

        # We use a dynamic style sheet variable for font size later
        self.base_stylesheet = """
                QFrame {{
                    background-color: rgba(50, 50, 50, 150);
                    border-radius: {radius}px;
                    border: 1px solid rgba(100, 100, 100, 100);
                }}
                QToolButton {{
                    background-color: transparent;
                    color: white;
                    border: none;
                    font-weight: bold;
                    padding: {padding}px;
                    font-size: {fontsize}px;
                }}
                QToolButton:hover {{
                    background-color: rgba(255, 255, 255, 50);
                    border-radius: 3px;
                }}
                QToolButton:checked {{
                    background-color: rgba(0, 200, 255, 100);
                    border: 1px solid #00c8ff;
                }}
                QLabel {{
                    color: white; 
                    border: none; 
                    font-size: {fontsize}px;
                }}
            """

        layout = QHBoxLayout(self.overlay_frame)
        # Margins/Spacing will be set dynamically in resizeEvent
        self.toolbar_layout = layout

        # --- 2. Add Existing Buttons ---

        # Draw Button
        btn_draw = QToolButton()
        btn_draw.setText("Draw / Cut")
        btn_draw.setDefaultAction(self.draw_action)
        btn_draw.setCheckable(True)
        layout.addWidget(btn_draw)

        # Reset Button
        btn_clear = QToolButton()
        btn_clear.setText("Reset")
        btn_clear.clicked.connect(self.showTotal)
        layout.addWidget(btn_clear)

        # Snap Button
        btn_snap = QToolButton()
        btn_snap.setText("Snap")
        btn_snap.clicked.connect(self.take_screenshot)
        layout.addWidget(btn_snap)

        # --- 3. Add Orbit Buttons ---

        # Add a visual separator (optional)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("background-color: rgba(255,255,255,50);")
        layout.addWidget(line)

        # Left Orbit
        btn_left = QToolButton()
        btn_left.setText("←")  # Unicode Arrow
        btn_left.setToolTip("Hold to Rotate Left")

        btn_left.setAutoRepeat(True)
        btn_left.setAutoRepeatDelay(300)  # Wait 300ms before starting to repeat
        btn_left.setAutoRepeatInterval(20)  # Repeat every 20ms (~50 FPS)


        # Connect PRESSED to Start, RELEASED to Stop
        btn_left.pressed.connect(self.start_orbit_left)
        #btn_left.released.connect(self.stop_orbit)
        layout.addWidget(btn_left)

        # Right Orbit
        btn_right = QToolButton()
        btn_right.setText("→")  # Unicode Arrow
        btn_right.setToolTip("Hold to Rotate Right")

        btn_left.setAutoRepeat(True)
        btn_left.setAutoRepeatDelay(300)  # Wait 300ms before starting to repeat
        btn_left.setAutoRepeatInterval(20)  # Repeat every 20ms (~50 FPS)

        btn_right.pressed.connect(self.start_orbit_right)
        #btn_right.released.connect(self.stop_orbit)
        layout.addWidget(btn_right)

        # --- 4. Add Slider ---

        self.lbl_thresh = QLabel("Thresh:")
        layout.addWidget(self.lbl_thresh)

        self._verticalSlider_1.setOrientation(Qt.Horizontal)
        layout.addWidget(self._verticalSlider_1)

        self.overlay_frame.show()

        # Trigger initial sizing
        self.update_toolbar_size()

    def update_toolbar_size(self):
        """
        Calculates size proportional to the screen/widget width.
        """
        if not hasattr(self, 'overlay_frame'):
            return

        # Calculate scale factor based on widget width (e.g., standard width 800)
        # You can tweak the '800' to be whatever your "standard" screen width is.
        scale = max(0.6, min(2.0, self.width() / 300.0))

        # Calculate Dynamic Metrics
        font_size = int(12 * scale)
        padding = int(4 * scale)
        radius = int(5 * scale)
        slider_width = int(100 * scale)
        margin = int(5 * scale)

        # Apply Stylesheet
        style = self.base_stylesheet.format(
            fontsize=font_size,
            padding=padding,
            radius=radius
        )
        self.overlay_frame.setStyleSheet(style)

        # Apply Layout Spacing
        self.toolbar_layout.setContentsMargins(margin, margin, margin, margin)
        self.toolbar_layout.setSpacing(margin)

        # Apply Component Specifics
        self._verticalSlider_1.setFixedWidth(slider_width)

        # Resize and Reposition
        self.overlay_frame.adjustSize()

        # Position: Top-Center (or keep it 10,10 if you prefer)
        # Center X = (Window Width - Toolbar Width) / 2
        # x_pos = (self.width() - self.overlay_frame.width()) // 2

        # Position: Proportional Margin from Top-Left
        x_pos = int(10 * scale)
        y_pos = int(10 * scale)

        self.overlay_frame.move(x_pos, y_pos)

    def resizeEvent(self, event):
        """
        Override the built-in resize event to update toolbar size/position.
        """
        self.update_toolbar_size()
        # Don't forget to call the super class resize event if needed
        super().resizeEvent(event)


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

        opts = QFileDialog.DontUseNativeDialog
        filter_str = (
            "Standard Screen Res (*.png);;"
            "High Res (1000px) (*.png);;"
            "High Res (2000px) (*.png);;"
            "High Res (3000px) (*.png)"
        )

        # Define exactly which one you want as the default
        default_filter = "Standard Screen Res (*.png)"

        opts = QFileDialog.DontUseNativeDialog

        fileObj, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save File",
            self.source_dir,
            filter=filter_str,
            initialFilter=default_filter,  # This sets the primary selection
            options=opts
        )
        if fileObj == '':
            return
        filename = fileObj + '.png'
        init_width, init_height = self.width(), self.height()
        maxhw = max(init_width, init_height)
        width, height = init_width, init_height

        if "1000px" in selected_filter:
            scale = 3000 / max(init_width, init_height)
            width = int(init_width * scale)
            height = int(init_width * scale)

        elif "2000px" in selected_filter:
            scale = 6000 / max(init_width, init_height)
            width = int(init_width * scale)
            height = int(init_width * scale)
        elif "3000px" in selected_filter:
            scale = 6000 / max(init_width, init_height)
            width = int(init_width * scale)
            height = int(init_width * scale)
        else:
            width = width+1
            height = height+1

        self.setFixedWidth(int(width))
        self.setFixedHeight(int(height))

        # Force the OS to redraw the window at the new size
        # You might need to call this multiple times or use a small sleep
        QApplication.processEvents()
        self.makeCurrent()
        self.paintGL()
        glFinish()  # Wait for GPU to finish

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
        QApplication.processEvents()
        self.makeCurrent()
        self.paintGL()
        glFinish()  # Wait for GPU to finish



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
        Optimized 'Carving': Removes 3D volume parts inside the drawn polygon.
        Uses fast bounding-box depth reading + vectorized unprojection.
        """
        if len(self.totalpolys[0]) < 3:
            return

        # 1. Setup Polygon & Bounding Box
        # Convert points to integer array for OpenCV
        pts = np.array(self._sPoints, dtype=np.int32)

        # Get Bounding Box (x, y, width, height)
        bx, by, bw, bh = cv2.boundingRect(pts)
        if bw <= 0 or bh <= 0: return

        # 2. Optimized Depth Read (Bounding Box Only)
        # We replace the full-screen 'self._z' with a tiny 'z_crop'
        self.makeCurrent()

        # Note: OpenGL Y is inverted (bottom-up) vs Qt (top-down)
        gl_y_bottom = self.height() - (by + bh)

        # Read only the depth inside the bounding box
        z_crop_raw = glReadPixels(bx, gl_y_bottom, bw, bh, GL_DEPTH_COMPONENT, GL_FLOAT)
        z_crop = np.frombuffer(z_crop_raw, dtype=np.float32)

        # Find Z range (Depth bounds)
        # Filter out 1.0 (background) and 0.0 (too close)
        valid_z = z_crop[(z_crop > 0.0) & (z_crop < 1.0)]

        if valid_z.size == 0:
            # Fallback if clicking empty space: carve through whole volume
            z_min, z_max = 0.0, 1.0
        else:
            # Add a tiny buffer to ensure we cover the edges
            z_min = valid_z.min() - 0.001
            z_max = valid_z.max() + 0.001

        # 3. Create 2D Mask (Instant rasterization with OpenCV)
        # This replaces the slow 'filledPoly' / matplotlib logic
        mask_2d = np.zeros((self.height(), self.width()), dtype=np.uint8)
        cv2.fillPoly(mask_2d, [pts], 1)

        # Get indices of all pixels inside the polygon
        # ys, xs are global window coordinates
        ys, xs = np.where(mask_2d > 0)
        if len(xs) == 0: return

        # 4. Vectorized Ray Casting (The Speedup)
        # Instead of looping Z in python, we create a massive coordinate array

        # How many Z-steps to take through the object?
        # (Higher = cleaner cut, Lower = faster)
        # 1. Find the center of the 2D bounding box (NDC)
        # We only need one ray to estimate the depth distance
        center_x_ndc = (2.0 * (bx + bw / 2.0) / self.width()) - 1.0
        center_y_ndc = 1.0 - (2.0 * (by + bh / 2.0) / self.height())

        # 2. Create Clip Coordinates for Near and Far points of this ray
        # Point A at z_min, Point B at z_max
        # Z in clip space = 2*z - 1
        clip_near = np.array([center_x_ndc, center_y_ndc, 2.0 * z_min - 1.0, 1.0])
        clip_far = np.array([center_x_ndc, center_y_ndc, 2.0 * z_max - 1.0, 1.0])

        # 3. Unproject to World Space (Voxel Coordinates)
        invM = np.linalg.inv(self.mvpProj.reshape((4, 4)).T)

        def unproject_single(v, matrix):
            w = np.dot(matrix, v)
            if w[3] != 0: w /= w[3]
            return w[:3]  # Return XYZ

        world_near = unproject_single(clip_near, invM)
        world_far = unproject_single(clip_far, invM)

        # 4. Calculate Distance in Voxels
        # Since your world coordinates map to voxels, this is the exact depth in voxels
        voxel_dist = np.linalg.norm(world_far - world_near)

        # 5. Set Steps Automatically
        # Sampling rate of 1.5 to 2.0 is safe (Nyquist theorem-ish)
        # Clamp to reasonable limits (e.g. at least 10 steps)
        num_z_steps = int(max(10, voxel_dist * 2.0))
        #num_z_steps = min(num_z_steps, 200)
        print(num_z_steps)
        z_vals = np.linspace(z_min, z_max, num_z_steps)

        # Convert Pixels to Normalized Device Coordinates (NDC)
        # X: 0..Width -> -1..1
        ndc_xs = (2.0 * xs / self.width()) - 1.0
        # Y: 0..Height -> 1..-1 (Flip Y)
        ndc_ys = 1.0 - (2.0 * ys / self.height())

        # Create the grid of points to unproject
        # We repeat the (X,Y) for every Z step
        all_xs = np.repeat(ndc_xs, num_z_steps)
        all_ys = np.repeat(ndc_ys, num_z_steps)
        # Tile Z values: z1, z2, z3... z1, z2, z3...
        all_zs = np.tile(2.0 * z_vals - 1.0, len(xs))  # Clip space Z (-1..1)

        # Stack into (4, N) matrix [x, y, z, w]
        ones = np.ones_like(all_xs)
        clip_coords = np.vstack([all_xs, all_ys, all_zs, ones])

        # 5. Inverse Projection (Matrix Multiplication)
        # Convert Clip Space -> World Space
        invM = np.linalg.inv(self.mvpProj.reshape((4, 4)).T)
        world_coords = np.dot(invM, clip_coords)

        # Perspective Divide (XYZ / W)
        # Avoid divide by zero
        w_comp = world_coords[3, :]
        w_comp[w_comp == 0] = 1.0
        world_coords /= w_comp

        # 6. Map World Coordinates to Voxel Indices
        # Extract raw World X, Y, Z
        wx = world_coords[0, :]
        wy = world_coords[1, :]
        wz = world_coords[2, :]

        # Apply your specific coordinate transformations
        # Based on your original code:
        # selected[:, 2] = self.maxXYZ[0] - selected[:, 2] -> Voxel X from World Z
        # selected[:, 0] = self.maxXYZ[2] - selected[:, 0] -> Voxel Z from World X
        # selected[:, 1] -> Voxel Y from World Y

        # Map to integers
        vox_x = (self.maxXYZ[0] - wz).astype(np.int32)
        vox_y = (wy).astype(np.int32)
        vox_z = (self.maxXYZ[2] - wx).astype(np.int32)

        # 7. Filter Valid Indices
        xmax, ymax, zmax = self.maxCoord

        valid_mask = (
                (vox_x >= 0) & (vox_x < xmax) &
                (vox_y >= 0) & (vox_y < ymax) &
                (vox_z >= 0) & (vox_z < zmax)
        )

        # Keep only valid voxel indices
        vx = vox_x[valid_mask]
        vy = vox_y[valid_mask]
        vz = vox_z[valid_mask]

        if vx.size > 0:
            # 8. Update The Masks
            # Update the boolean exclusion mask
            # Note: Your logic used indices [0, 1, 2] which corresponds to our [vx, vy, vz]
            self._excluded_inds[vx, vy, vz] = True

            # Recalculate Indices Mask (Optional, matches your logic)
            # self._indices = ... (Update if needed based on _excluded_inds)

            # Zero out the image data
            self._seg_im[self._excluded_inds, :] = 0

            # 9. Upload to GPU
            self.GLV.setData(self._seg_im, self._artistic)
            self.GLV.update()

        # Cleanup
        self.clear_points()


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
        Paint GL and overlay text info.
        """
        # 1. Draw the 3D Scene (Super class handles the heavy lifting)
        super(glScientific, self).paintGL(region, viewport, useItemNames)

        # 2. Prepare Data to Display
        elev = self.opts['elevation']
        azim = self.opts['azimuth']
        dist = self.opts['distance']
        azim = azim % 360
        info_text = f"Elev: {elev:.1f}\nAzim: {azim:.1f}\nDist: {dist:.1f}"

        # 3. Draw 2D Text Overlay
        # We use QPainter directly on the widget surface
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)

        # Setup Font
        font = QFont("Arial", 10)  # Size 10
        font.setBold(True)
        painter.setFont(font)

        # Setup Color (e.g., Bright Yellow for visibility)
        painter.setPen(QColor(255, 255, 0))
        # Define proportions (e.g., box is 20% of width, 10% of height)
        box_width = int(self.width() * 0.12)
        box_height = int(self.height() * 0.06)

        # Ensure minimum readable size (optional safety check)
        box_width = max(box_width, 20)
        box_height = max(box_height, 10)

        # Define padding as 2% of width
        padding_x = int(self.width() * 0.02)
        padding_y = int(self.height() * 0.02)

        # Calculate X and Y for Lower Right
        # x = Width - BoxWidth - Padding
        rect_x = self.width() - box_width - padding_x
        # y = Height - BoxHeight - Padding
        rect_y = self.height() - box_height - padding_y

        rect = QRect(rect_x, rect_y, box_width, box_height)

        painter.drawText(rect, Qt.AlignLeft | Qt.AlignTop, info_text)

        painter.end()

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

    def mouseReleaseEvent(self, ev):
        """
        Optimized Mouse Release:
        1. Closes the polygon loop.
        2. Executes the 'cut' (remove_painting) efficiently.
        3. Swaps Scatter plot back to Volume rendering without double-uploading.
        """
        # Flag to prevent double-uploading data
        volume_data_updated = False

        # --- 1. Polygon / Carving Logic ---
        if self._enabledPolygon:
            # Need at least 3 points to make a polygon
            if len(self._sPoints) < 3:
                self._sPoints = []
                #self._enabledPolygon = False
                #self.draw_action.setChecked(False)
                return

            # A. Close the loop (Connect last point to first)
            # Assuming _numPolys corresponds to the current active polygon
            first_point = self.totalpolys[self._numPolys][0]
            self.totalpolys[self._numPolys].append(first_point)
            self._sPoints.append(self._sPoints[0])

            # B. Store Camera Meta-data (Cleaner dictionary lookup)
            axis_map = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
            # Use getattr to be safe if _ax or _d are not initialized
            axis_name = axis_map.get(getattr(self.GLV, '_ax', 0), 'sagittal')
            direction = getattr(self.GLV, '_d', 1)

            self.totalpolys_camer[self._numPolys] = [axis_name, direction]

            # C. Perform the Cut (remove_painting)
            try:
                # This function calculates the cut AND uploads to GPU
                self.remove_painting()
                volume_data_updated = True  # Mark that GPU has fresh data
            except Exception as e:
                print(f"Error in remove_painting: {e}")

            # D. Cleanup / Reset State
            self._numPolys = 0  # Reset index

            # Update visual polygon item (if you want to keep the line visible)
            if hasattr(self, 'GLPl'):
                self.GLPl.update_points(self.totalpolys)

            self._sPoints = []
            #self._enabledPolygon = False
            #self.draw_action.setChecked(False)

        # --- 2. View Switching (Scatter -> Volume) ---
        # If we were using a Scatter plot (for fast interaction), switch back to Volume
        if 'scatter_total' in self.items:
            self.removeItem('scatter_total')

            # OPTIMIZATION: Only upload data if remove_painting didn't just do it.
            if not volume_data_updated:
                self.GLV.setData(self._seg_im, self._artistic)

            # Ensure Volume is visible and depth is correct
            self.GLV.setDepthValue(20)
            self.addItem(self.GLV, 'vol_total')

            # Force visual refresh
            self.update()

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

    def ray_cast_value(self, ev):
        """
        Manually shoots a ray from the mouse position into the dataset
        to find the first non-zero voxel (simulating a depth read).
        """
        # 1. Get the Ray Start (Near Plane) and End (Far Plane)
        # We use your existing helper to get World Coordinates at z=0 and z=1
        p_start, _ = self._compute_coordinates(ev, z=0.0, exact=True)
        p_end, _ = self._compute_coordinates(ev, z=1.0, exact=True)

        # Convert to numpy arrays for vector math
        p0 = np.array(p_start[:3])
        p1 = np.array(p_end[:3])

        # 2. Calculate Direction and Steps
        # Length of the ray through the world
        vector = p1 - p0
        length = np.linalg.norm(vector)
        if length == 0: return None, None

        direction = vector / length

        # Determine step size (0.5 ensures we don't skip voxels)
        step_size = 0.5
        # Limit the search distance (e.g., to the diagonal of the volume)
        max_dist = np.linalg.norm(self.maxXYZ) * 2.0

        # 3. March along the ray
        current_dist = 0.0

        # Get array bounds
        d_max, h_max, w_max = self._seg_im.shape[:3]

        while current_dist < max_dist:
            # Current World Coordinate
            p_curr = p0 + direction * current_dist

            # --- Apply Your Coordinate Transforms (World -> Voxel) ---
            # NOTE: These must match exactly the inverse of how you display them.
            # Based on your gotoLoc logic:
            # vox_z = max[0] - world_z
            # vox_x = max[2] - world_x
            # vox_y = world_y

            vox_z_idx = int(self.maxXYZ[0] - p_curr[2])
            vox_x_idx = int(self.maxXYZ[2] - p_curr[0])
            vox_y_idx = int(p_curr[1])

            # 4. Check Bounds
            if (0 <= vox_x_idx < d_max) and \
                    (0 <= vox_y_idx < h_max) and \
                    (0 <= vox_z_idx < w_max):

                # 5. Check Data Intensity
                # Accessing self._seg_im[x, y, z]
                # We check the Alpha channel (index 3) if it exists, or value > 0
                voxel_val = self._seg_im[vox_x_idx, vox_y_idx, vox_z_idx]

                # Assuming RGBA (4 channels) or Scalar
                hit = False
                if isinstance(voxel_val, np.ndarray) and voxel_val.shape[0] > 3:
                    # Check Alpha channel (transparency)
                    if voxel_val[3] > 0:
                        hit = True
                elif voxel_val > 0:
                    hit = True

                if hit:
                    # FOUND IT!
                    # Return the raw World Coordinate of the hit
                    return np.array([p_curr[0], p_curr[1], p_curr[2], 1.0])

            current_dist += step_size

        return None

    def _compute_coordinates(self, ev, z=None, exact=False):
        """
        Convert 2D Mouse Position + Depth (Z) -> 3D World Coordinates.
        """
        # 1. Handle Z-Depth (Depth buffer value 0.0 to 1.0)
        if z is None:
            # Fallback: Safe single-pixel read (Only if caller forgot to pass z)
            self.makeCurrent()
            z = glReadPixels(ev.x(), self.height() - ev.y(), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]


        # 2. Background Handling
        # If z is 1.0 (Background) or 0.0 (Near clip), use the last valid depth
        # This allows drawing "in the air" near the object.
        if not exact:
            if z == 1.0 or z == 0.0:
                z = self._lastZ
            else:
                self._lastZ = z

        # 3. Normalized Device Coordinates (NDC)
        # Convert x,y from Pixels (0 to Width) to NDC (-1 to +1)
        # OpenGL Y is bottom-up, Qt is top-down
        w = self.width()
        h = self.height()

        ndc_x = (2.0 * ev.x() / w) - 1.0
        ndc_y = 1.0 - (2.0 * ev.y() / h)

        # 4. Construct the Clip Space Vector
        # OpenGL Range for Z is -1.0 to 1.0 in Clip Space (Mapped from 0..1 depth)
        clip_z = 2.0 * z - 1.0
        clip_coords = np.array([ndc_x, ndc_y, clip_z, 1.0])

        # 5. Unproject (Clip Space -> World Space)
        # We use the Inverse ModelViewProjection Matrix
        # Note: If self.mvpProj is a Qt Matrix, use .inverted()[0]
        # If it is a numpy array, use np.linalg.inv
        try:
            if isinstance(self.mvpProj, np.ndarray):
                inv_mvp = np.linalg.inv(self.mvpProj.reshape((4, 4)).T)
            else:
                # Assuming flattened list or similar structure
                inv_mvp = np.linalg.inv(np.array(self.mvpProj).reshape((4, 4)).T)

            world_coords = np.dot(inv_mvp, clip_coords)

        except np.linalg.LinAlgError:
            return [0, 0, 0, 0], [ev.x(), ev.y()]

        # 6. Perspective Divide (Homogeneous -> Cartesian)
        if world_coords[3] != 0:
            world_coords /= world_coords[3]

        # Return 3D point (x,y,z,w) and original 2D mouse pos
        return world_coords, [ev.x(), ev.y()]

    def mouseMoveEvent(self, ev):
        if not hasattr(self, 'mousePos'):
            return

        # Current mouse position
        curr_pos = ev.pos()
        diff = curr_pos - self.mousePos
        self.mousePos = curr_pos

        if self._enabledPolygon:
            # 1. Safety Check: Ensure we have a started polygon
            if self._numPolys not in self.totalpolys:
                return

            # 2. OPTIMIZATION: Distance Threshold Check
            # Only add a point if we moved at least 5 pixels from the last ADDED point.
            # You need to store 'last_added_pos' in mousePressEvent initially.
            if hasattr(self, 'last_added_pos'):
                dist = (curr_pos - self.last_added_pos).manhattanLength()
                if dist < 5:  # 5 pixel threshold (adjust for smoothness vs precision)
                    return

                    # Update the last added position
            self.last_added_pos = curr_pos

            # 3. Calculate 3D Coordinates
            # Use z=0.0 (Near Plane) for the VISUAL polygon so it appears on top
            # 1. Activate Context & Read "True" Surface Depth
            #self.makeCurrent()
            #y_gl = self.height() - ev.y()
            #z_surface = glReadPixels(ev.x(), y_gl, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]

            # 2. Apply Bias (Move closer to camera)
            # 0.0 is Near (Camera), 1.0 is Far (Background).
            # Subtracting 0.005 moves the point slightly in front of the object.
            # We use max(0, ...) to ensure we don't go behind the camera.
            #z_visual = max(0.0, z_surface - 0.005)

            # 3. Compute VISUAL points using the Biased Z
            points, _ = self._compute_coordinates(ev, z=None)
            p1, xo = self._compute_coordinates(ev, z=1)  # xo is raw coordinate for your cutter

            x, y, z, _ = points
            if z == 0: return

            # 4. Boundary Check
            endIm = self._endIMage(x, y, z)
            if endIm[0]:
                x, y, z = endIm[1], endIm[2], endIm[3]

            # 5. Append Point (Lasso Logic)
            self._sPoints.append(xo)
            self.totalpolys[self._numPolys].append([x, y, z])

            # 6. Update Visuals (Only if we have enough points)
            if len(self.totalpolys[self._numPolys]) > 1:
                self.GLPl.update_points(self.totalpolys)

                # Check if item exists to avoid redundant addItem calls (Performance)
                if 'pol_total' not in self.items:
                    self.addItem(self.GLPl, 'pol_total')
                    self.GLPl.setDepthValue(100)

        # --- Navigation Logic (Standard) ---
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

            self.update()

    def emit_viewport(self, points):
        self.point3dpos.emit(points, None)


    def leaveEvent(self, a0):
        if hasattr(self, 'orbit_timer') and self.orbit_timer.isActive():
            self.orbit_timer.stop()

    def mousePressEvent(self, ev):
        """
        By Bahram Jafrasteh
        :param ev:
        :return:
        """
        #self.opts['azimuth'] = 0
        #self.opts['elevation'] = 0
        if hasattr(self, 'orbit_timer') and self.orbit_timer.isActive():
            self.orbit_timer.stop()
        self.mousePos = ev.localPos()
        if not (self._gotoLoc or self._enabledPolygon):
            return

        if ev.button()==Qt.RightButton:
            return
        if 'vol_total' in self.items:
            self.SubUpdateSegScatterItem()
            # --- OPTIMIZATION START ---
            # 2. Activate Context & Read Depth
            # We read the depth of the pixel user clicked on immediately.
            # This captures the Z-value of the object *before* any scene updates occur.
            self.makeCurrent()

            # Note: OpenGL Y-axis is bottom-up, Qt is top-down. We flip Y.
            # We use ev.x() and ev.y() for integer pixel coordinates.
            y_gl = self.height() - ev.y()
            z_val = glReadPixels(ev.x(), y_gl, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
            self._z = z_val
            # --- OPTIMIZATION END ---
        else:
            return
        # 3. Pass 'z_val' to compute_coordinates
        # This prevents the function from trying to re-read the screen or call showEvent(0)
        points, _ = self._compute_coordinates(ev, z=z_val)

        #points[1] -= 1
        #print(points)

        if self._enabledPolygon:
            # Calculate 'p1' (Far plane, z=1) and 'xo' (Near plane/Start)
            # We pass z=1 explicitely here too, which is efficient.
            self.last_added_pos = ev.pos()
            p1, xo = self._compute_coordinates(ev, z=1)

            self._sPoints = []
            self.totalpolys[self._numPolys] = []

            if points[-2] == 0:  # Check for W component or invalid div
                return

            x, y, z, _ = points

            # Check bounds
            endIm = self._endIMage(x, y, z)
            if endIm[0]:
                x, y, z = endIm[1], endIm[2], endIm[3]

            # Add points
            self._sPoints.append(xo)
            self.totalpolys[self._numPolys].append([x, y, z])
            # ADD THIS: Add a duplicate "ghost" point for the mouse to drag
            self._sPoints.append(xo)
            self.totalpolys[self._numPolys].append([x, y, z])

            if len(self.totalpolys[self._numPolys]) > 1:
                self.GLPl.update_points(self.totalpolys)



        #print(points)
                # --- GOTO LOC Logic (Updated) ---
        elif self._gotoLoc:
            points = self.ray_cast_value(ev)

            if points is None:
                # Ray missed the volume (clicked background)
                return

            # 2. Apply Coordinate System Transformations
            # Convert OpenGL World coordinates back to your Voxel/Array coordinates.
            # (Based on the logic you provided)

            # Invert Z (Voxel X comes from World Z)
            points[2] = self.maxXYZ[0] - points[2]

            # Invert X (Voxel Z comes from World X)
            points[0] = self.maxXYZ[2] - points[0]

            # Note: points[1] (Y) usually maps directly to Voxel Y, unless flipped.
            # If you need Y flipping, uncomment: points[1] = self.maxXYZ[1] - points[1]

            # 3. Determine Axis Name for the signal
            windowName = 'sagittal'  # Default
            if hasattr(self.GLV, '_ax'):
                if self.GLV._ax == 0:
                    windowName = 'sagittal'
                elif self.GLV._ax == 1:
                    windowName = 'coronal'
                elif self.GLV._ax == 2:
                    windowName = 'axial'

            # 4. Emit the signal with the calculated surface point
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
            #self.GLV.setData(self._seg_im.copy(), self._artistic)
            self.GLV.setData(self._seg_im, self._artistic)
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
            self._enabledPolygon = value

            if not value:
                # --- Disable Mode ---
                self.removeItem('pol_total')
                self.totalpolys = defaultdict(list)
                # self.totalpolys_camer = defaultdict(list) # If needed
                if hasattr(self, 'GLPl'):
                    self.GLPl.update_points(self.totalpolys)

                # Restore Depth if needed
                self.GLV.setDepthValue(10)  # Restore to normal (e.g., 10 or 20)

            else:
                # --- Enable Mode ---
                # 1. Update Visuals
                self.SubUpdateSegScatterItem()

                # 2. Ensure Volume is visible and depth is set for drawing on top
                if 'scatter_total' in self.items:
                    self.removeItem('scatter_total')

                self.GLV.setData(self._seg_im, self._artistic)
                self.GLV.setDepthValue(0)  # Bring to front for drawing
                self.GLPl.setDepthValue(100)
                self.addItem(self.GLV, 'vol_total')

                # 3. Force a clean update (Visual only)
                self.update()

                # NOTE: We removed glReadPixels from here.
                # We will read the depth in mousePressEvent instead.

        except Exception as e:
            print(f"Error in draw_status: {e}")

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
        if _image.ndim==3:
            if map_type=='original':
                cm = np.repeat(_image[...,None], 4, -1)
            else:
                import matplotlib.pyplot as plt
                cm = plt.get_cmap(map_type, lut=256)
                cm = cm(_image / _image.max()) * 255.0
                #if map_type!='gist_rainbow_r':
                #    cm[...,3]=self._image
        elif _image.ndim == 4:
            cm = _image.copy()

            # 1. Calculate 3D Intensity Map from 4D data
            # We take the max across the last axis (channels) to see if ANY channel has data
            intensity_map = _image.max(axis=-1)

            # 2. Create the 3D Mask
            # Check if the brightest channel in the voxel is below the threshold
            threshold_val = intensity_map.max() * self._threshold / 100.0
            mask = intensity_map <= threshold_val

            # 3. Apply Coloring (Your logic)
            # Note: Your original loop for map_type != 'original' might fail because
            # cmap() returns 4 values (RGBA), but you are assigning it to 1 channel.
            # Assuming standard behavior here:
            if map_type != 'original':
                # If you really need to recolor specific channels, do it here.
                # Otherwise, 4D input is usually already colored (RGBA).
                pass

                # 4. Apply the Mask to the Alpha Channel
            # Ensure cm has an alpha channel (4 channels)
            if cm.shape[-1] == 4:
                # Set Alpha to 0 (Transparent) where mask is True
                cm[mask, 3] = 0.0
            else:
                # add one channel for alpha
                alpha_channel = np.ones(cm.shape[:-1] + (1,), dtype=cm.dtype,) * 255
                alpha_channel[mask] = 0.0
                cm = np.concatenate([cm, alpha_channel], axis=-1)

            # 5. Handle Exclusion (if you have the excluded_inds logic from previous steps)
            if hasattr(self, '_excluded_inds') and self._excluded_inds is not None:
                cm[self._excluded_inds, 3] = 0.0
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

    def updateSegVolItem(self, imSeg=None, imOrg=None, currentWidnowName=None, sliceNum=None):


        if imSeg is None: return

        # 1. OPTIMIZATION: Fast Sparse Check
        # If the segmentation is empty, don't allocate massive arrays
        active_indices = np.where(imSeg > 0)
        if active_indices[0].size <= 1:
            self.removeItem('scatter_total')
            return

        # Update Bounds (Keep your existing logic)
        self.maxXYZ_current = [active_indices[0].max(), active_indices[1].max(), active_indices[2].max()]
        self.minXYZ_current = [active_indices[0].min(), active_indices[1].min(), active_indices[2].min()]

        # 2. OPTIMIZATION: LUT Generation
        max_idx = int(imSeg.max())
        # Use uint8 for LUT immediately (saves 4x memory vs float32)
        lut = np.zeros((max_idx + 1, 4), dtype=np.uint8)
        show_all = 9876 in self.colorInds

        for label, color in self.colorsCombinations.items():
            if label > max_idx: continue
            if show_all or (label in self.colorInds):
                # Assume color is normalized 0-1, scale to 0-255 uint8
                lut[int(label)] = (np.array(color) * 255).astype(np.uint8)

        # 3. OPTIMIZATION: Apply LUT (Generate Segmentation Layer)
        # This creates the RGBA volume for segmentation
        seg_vol = lut[imSeg.astype(np.int32)]  # Shape: (H, W, D, 4) uint8

        # 4. OPTIMIZATION: Blending Logic
        # We avoid allocating a huge 'cm' array if possible
        if self.im_seg_action.isChecked() and imOrg is not None:

            # A. Create Output Volume (Pre-allocate)
            # We use the segmentation volume as the base to save one allocation
            final_vol = seg_vol  # Reference, not copy yet

            # B. Define Masks
            # Pixels where Segmentation exists
            mask_seg = seg_vol[..., 3] > 0

            # Pixels where Image (MRI/US) exists above threshold
            # Assuming imOrg is uint8 (0-255)
            # Calculate threshold once
            thresh_val = imOrg.max() * self._threshold / 100
            mask_img = imOrg > thresh_val

            # C. EFFICIENT BLENDING
            # We only blend where BOTH exist.
            # Where only Image exists -> Just copy Image.
            # Where only Seg exists -> Keep Seg.

            # Case 1: Only Background Image (No Seg)
            # Find pixels with Image BUT NO Segmentation
            # We manually construct RGBA for these pixels in place
            mask_only_img = mask_img & (~mask_seg)

            if np.any(mask_only_img):
                # Extract grayscale values
                gray_vals = imOrg[mask_only_img]

                # Apply Intensity scaling
                alpha_vals = (gray_vals.astype(np.float32) / 255.0) * self.intensityImg
                alpha_uint8 = (alpha_vals * 255).astype(np.uint8)

                # Assign to final volume [R, G, B, A]
                # Since it's grayscale, R=G=B=Value
                final_vol[mask_only_img, 0] = gray_vals
                final_vol[mask_only_img, 1] = gray_vals
                final_vol[mask_only_img, 2] = gray_vals
                final_vol[mask_only_img, 3] = alpha_uint8

            # Case 2: Blend Intersection (Image + Seg)
            mask_blend = mask_img & mask_seg

            if np.any(mask_blend):
                # This is the heavy part, but now we only run it on ~5% of pixels

                # Get Source (Seg)
                s_rgba = seg_vol[mask_blend].astype(np.float32) / 255.0
                s_rgb = s_rgba[:, :3]
                s_a = np.sqrt(s_rgba[:, 3])  # Non-linear boost

                # Get Dest (Image)
                d_val = imOrg[mask_blend].astype(np.float32) / 255.0
                d_rgb = d_val[:, None]  # Broadcast grayscale to RGB
                d_a = d_val * self.intensityImg

                # Alpha Composite (Standard Over Operator)
                out_a = s_a + d_a * (1 - s_a)
                safe_a = np.clip(out_a, 1e-6, 1.0)  # Prevent div/0

                term1 = s_rgb * s_a[:, None]
                term2 = d_rgb * d_a[:, None] * (1 - s_a[:, None])

                out_rgb = (term1 + term2) / safe_a[:, None]

                # Write Back
                blended_px = np.zeros((mask_blend.sum(), 4), dtype=np.uint8)
                blended_px[:, :3] = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)
                blended_px[:, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

                final_vol[mask_blend] = blended_px

            self._seg_im = final_vol

        else:
            # Just return the colored segmentation
            self._seg_im = seg_vol

        # 5. Upload to Viewer
        # Ensure artistic mode is handled
        self.GLV.setData(self._seg_im, self._artistic)
        self.GLV.setDepthValue(20)
        self.addItem(self.GLV, 'vol_total')



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
        self.v._excluded_inds = np.zeros_like(data, dtype=bool)
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



