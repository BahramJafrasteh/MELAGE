"""
    Main BrainExtractor class
"""
__AUTHOR__ = 'Bahram Jafrasteh'

import os
import warnings
import numpy as np
import nibabel as nib
import trimesh
from numba import jit, prange
from PyQt5 import QtCore, QtGui, QtWidgets

from numba.typed import List
from utils.brain_extraction_helper import sphere, closest_integer_point, bresenham3d, l2norm, l2normarray, diagonal_dot

from PyQt5.QtCore import pyqtSignal
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

def to_str(val):
    return '{:.2f}'.format(val)

class BET(QDialog):
    closeSig = pyqtSignal()
    betcomp = pyqtSignal()
    datachange = pyqtSignal()

    """
    https://github.com/vanandrew/brainextractor
    Implemenation of the FSL Brain Extraction Tool
Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568; PMCID: PMC6871816.
    This class takes in a Nifti1Image class and generates
    the brain surface and mask.
    """
    def __init__(self, parent=None
                 ):
        super(BET, self).__init__(parent)


        """
        Initialization of Brain Extractor

        Computes image range/thresholds and
        estimates the brain radius
        """
    def set_pars(self, t02t: float = 0.02,
                 t98t: float = 0.98,
                 bt: float = 0.5,
                 d1: float = 20.0,  # mm
                 d2: float = 10.0,  # mm
                 rmin: float = 3.33,  # mm
                 rmax: float = 10.0,# mm
                 n_iter = 1000):


        #print("Initializing...")
        self.ht_min = t02t
        self.ht_max = t98t
        self.ft = bt
        self.sd_min = d2
        self.sd_max = d1
        self.r_min = rmin
        self.r_max = rmax
        self.n_iter = n_iter
        # get image resolution
        # store brain extraction parameters
        self.setupUi()
    def setData(self, img, res):
        self.res = res
        # store the image
        self.img = img

        # store conveinent references
        self.data = img # 3D data
        self.rdata = img.ravel()  # flattened data
        self.shape = img.shape  # 3D shape
        self.rshape = np.multiply.reduce(img.shape)  # flattened shape

    def get_params(self):
        state = self.checkbox_thresholding.isChecked()
        if state:
            from utils.utils import Threshold_MultiOtsu
            numc = int(self.histogram_threshold_min.value())
            thresholds = Threshold_MultiOtsu(self.img, numc)
            t02t = thresholds[0]
            t98t = thresholds[-1]
        else:
            t02t = self.histogram_threshold_min.value()/100
            t98t = self.histogram_threshold_max.value()/100


        bt = self.fractional_threshold.value()/100
        d1 = self.search_distance_max.value()
        d2 = self.search_distance_min.value()
        rmin = self.radcurv_min.value()
        rmax = self.radcurv_max.value()
        n_iter = self.iteration.value()
        return [t02t, t98t, bt, d1, d2, rmin, rmax,n_iter]
    def update_params(self):
        [t02t, t98t, bt, d1, d2, rmin, rmax,n_iter] = self.get_params()
        # store brain extraction parameters
        #print("Parameters: bt=%f, d1=%f, d2=%f, rmin=%f, rmax=%f" % (bt, d1, d2, rmin, rmax))
        self.bt = bt
        self.d1 = d1 / self.res
        self.d2 = d2 / self.res
        self.rmin = rmin / self.res
        self.rmax = rmax / self.res

        # compute E, F constants
        self.E = (1.0 / rmin + 1.0 / rmax) / 2.0
        self.F = 6.0 / (1.0 / rmin - 1.0 / rmax)


        self.n_iter = n_iter
        # get thresholds from histogram
        sorted_data = np.sort(self.rdata)
        self.tmin = np.min(sorted_data)
        state = self.checkbox_thresholding.isChecked()
        if state:
            self.t2 = t02t
            self.t98 = t98t
        else:
            self.t2 = sorted_data[np.ceil(t02t * self.rshape).astype(np.int64) + 1]
            self.t98 = sorted_data[np.ceil(t98t * self.rshape).astype(np.int64) + 1]
        self.tmax = np.max(sorted_data)
        self.t = (self.t98 - self.t2) * 0.1 + self.t2 #brain/background threshold
        #print("tmin: %f, t2: %f, t: %f, t98: %f, tmax: %f" % (self.tmin, self.t2, self.t, self.t98, self.tmax))

    def activate_advanced(self, value):
        self.widget.setEnabled(value)

    def setupUi(self):
        Dialog = self.window()
        Dialog.setObjectName("N4")
        Dialog.resize(310, 220)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0, 0, 0, 0)
        self.grid_main.setObjectName("gridLayout")

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setObjectName("gridLayout")
        self.grid_main.addLayout(self.hbox, 0, 0)

        self.checkBox = QtWidgets.QCheckBox()
        self.hbox.addWidget(self.checkBox, 0)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.stateChanged.connect(self.activate_advanced)
        self.comboBox_image = QtWidgets.QComboBox()
        self.comboBox_image.setGeometry(QtCore.QRect(20, 10, 321, 25))
        self.comboBox_image.setObjectName("comboBox")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        self.comboBox_image.currentIndexChanged.connect(self.datachange)
        self.hbox.addWidget(self.comboBox_image, 1)
        self.pushButton = QtWidgets.QPushButton()

        self.pushButton.setObjectName("pushButton")
        self.pushButton.pressed.connect(self.accepted_emit)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setMaximum(100)



        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.setContentsMargins(0, 0, 0, 0)
        self.hbox2.setObjectName("gridLayout")
        self.hbox2.addWidget(self.progressBar, 1)
        self.hbox2.addWidget(self.pushButton, 0)


        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter_3 = QtWidgets.QSplitter(self.widget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.label_2 = QtWidgets.QLabel(self.splitter_3)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.iteration = QtWidgets.QSpinBox(self.splitter_3)
        self.iteration.setObjectName("iteration")

        self.iteration.setMaximum(10000)
        self.iteration.setMinimum(1)
        self.iteration.setValue(self.n_iter)
        self.gridLayout.addWidget(self.splitter_3, 0, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.checkbox_thresholding = QtWidgets.QCheckBox(self.splitter)
        #self.checkbox_thresholding.setAlignment(QtCore.Qt.AlignCenter)
        self.checkbox_thresholding.setObjectName("checkbox_thresholding")
        self.checkbox_thresholding.setChecked(True)
        self.checkbox_thresholding.stateChanged.connect(self.change_hist_threshold)

        self.histogram_threshold_min = QtWidgets.QDoubleSpinBox(self.splitter)
        self.histogram_threshold_min.setObjectName("histogram_threshold_min")
        self.histogram_threshold_min.setValue(6)#(self.ht_min)*100)
        self.histogram_threshold_min.setMaximum(10)
        self.histogram_threshold_min.setMinimum(0)

        self.histogram_threshold_max = QtWidgets.QDoubleSpinBox(self.splitter)
        self.histogram_threshold_max.setObjectName("histogram_threshold_max")
        self.histogram_threshold_max.setValue((self.ht_max) * 100)
        self.histogram_threshold_max.setMaximum(100)
        self.histogram_threshold_max.setMinimum(0)

        self.histogram_threshold_min.setEnabled(True)
        self.histogram_threshold_max.setEnabled(False)

        self.gridLayout.addWidget(self.splitter, 1, 0, 1, 1)
        self.splitter_2 = QtWidgets.QSplitter(self.widget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.label = QtWidgets.QLabel(self.splitter_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.fractional_threshold = QtWidgets.QDoubleSpinBox(self.splitter_2)
        self.fractional_threshold.setObjectName("fractional_threshold")
        self.fractional_threshold.setValue((self.ft) * 100)
        self.fractional_threshold.setMaximum(100)
        self.fractional_threshold.setMinimum(0)
        self.gridLayout.addWidget(self.splitter_2, 2, 0, 1, 1)
        self.splitter_4 = QtWidgets.QSplitter(self.widget)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.label_4 = QtWidgets.QLabel(self.splitter_4)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.search_distance_min = QtWidgets.QDoubleSpinBox(self.splitter_4)
        self.search_distance_min.setObjectName("search_distance_min")
        self.search_distance_min.setValue(self.sd_min)
        self.search_distance_min.setMaximum(10000)
        self.search_distance_min.setMinimum(0)
        self.search_distance_max = QtWidgets.QDoubleSpinBox(self.splitter_4)
        self.search_distance_max.setObjectName("search_distance_max")
        self.search_distance_max.setValue(self.sd_max)
        self.search_distance_max.setMaximum(10000)
        self.search_distance_max.setMinimum(0)
        self.gridLayout.addWidget(self.splitter_4, 3, 0, 1, 1)
        self.splitter_5 = QtWidgets.QSplitter(self.widget)
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.label_5 = QtWidgets.QLabel(self.splitter_5)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.radcurv_min = QtWidgets.QDoubleSpinBox(self.splitter_5)
        self.radcurv_min.setObjectName("radcurv_min")
        self.radcurv_min.setValue(self.r_min)
        self.radcurv_min.setMaximum(10000)
        self.radcurv_min.setMinimum(0)
        self.radcurv_max = QtWidgets.QDoubleSpinBox(self.splitter_5)
        self.radcurv_max.setObjectName("radcurv_max")
        self.radcurv_max.setValue(self.r_max)
        self.radcurv_max.setMaximum(10000)
        self.radcurv_max.setMinimum(0)
        self.widget.setEnabled(False)
        self.gridLayout.addWidget(self.splitter_5, 4, 0, 1, 1)

        self.grid_main.addWidget(self.widget)
        self.grid_main.addLayout(self.hbox2, 20, 0)

        self.label_pr = QtWidgets.QLabel()
        self.label_pr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_pr.setObjectName("label_2")
        self.label_pr.setText('fdfdf')
        self.label_pr.setVisible(False)

        self.grid_main.addWidget(self.label_pr)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def change_hist_threshold(self, val):
        value = not val
        if not val:
            self.histogram_threshold_max.setEnabled(value)
            self.histogram_threshold_min.setMaximum(100)
            self.histogram_threshold_min.setMinimum(0)
            self.histogram_threshold_min.setValue(2)
        else:
            self.histogram_threshold_max.setEnabled(value)
            vl = self.histogram_threshold_min.value()
            if vl>10:
                self.histogram_threshold_min.setValue(10)
            elif vl<4:
                self.histogram_threshold_min.setValue(6)
            self.histogram_threshold_min.setMaximum(10)
            self.histogram_threshold_min.setMinimum(4)


    def clear(self):
        self.res = None
        # store the image
        self.img = None

        # store conveinent references
        self.data = None # 3D data
        self.rdata = None  # flattened data
        self.shape = None  # 3D shape
        self.rshape = None  # flattened shape
        self.mask = None

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "BET"))
        self.checkBox.setText(_translate("Dialog", "Advanced"))
        self.comboBox_image.setItemText(0, _translate("Dialog", "Top Image"))
        self.comboBox_image.setItemText(1, _translate("Dialog", "Bottom Image"))
        self.pushButton.setText(_translate("Dialog", "Apply"))
        self.label_2.setText(_translate("Dialog", "Iterations"))
        self.checkbox_thresholding.setText(_translate("Dialog", "Adaptive Thresholding"))
        self.label.setText(_translate("Dialog", "Fractional Threshold"))
        self.label_4.setText(_translate("Dialog", "Search Distance (mm)"))
        self.label_5.setText(_translate("Dialog", "Radius of curvature"))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(BET, self).closeEvent(a0)

    def accepted_emit(self):
        try:
            self.label_pr.setVisible(True)
            self.label_pr.setText('Initialization...')
            self.progressBar.setValue(0)
            self.update_params()
            self.progressBar.setValue(5)
            self._progress = 5
            self.initialization()
            self.label_pr.setText('preparation...')
            self.run()
            self.progressBar.setValue(98)
            self.label_pr.setText('Computing mask...')
            self.mask = self.compute_mask()
            self.label_pr.setVisible(False)
            self._progress = 100
            self.progressBar.setValue(self._progress)
            self._progress =0
            self.betcomp.emit()
        except Exception as e:
            print(e)

    def initialization(self):
        pv = self._progress
        self.progressBar.setValue(pv+1)
        # find the center of mass of image
        ic, jc, kc = np.meshgrid(
            np.arange(self.shape[0]), np.arange(self.shape[1]), np.arange(self.shape[2]), indexing="ij", copy=False
        )

        cdata = np.clip(self.rdata, self.t2, self.t98) * (self.rdata > self.t)
        ci = np.average(ic.ravel(), weights=cdata)
        cj = np.average(jc.ravel(), weights=cdata)
        ck = np.average(kc.ravel(), weights=cdata)
        self.c = np.array([ci, cj, ck])
        #print("Center-of-Mass: {}".format(self.c))

        # compute 1/2 head radius with spherical formula
        self.r = 0.5 * np.cbrt(3 * np.sum(self.rdata > self.t) / (4 * np.pi))
        #print("Head Radius: %f" % (2 * self.r))

        # get median value within estimated head sphere
        self.tm = np.median(self.data[sphere(self.shape, 2 * self.r, self.c)]) #media of all point within a sphere of the estimated radius
        #print("Median Intensity within Head Radius: %f" % self.tm)

        # generate initial surface
        #print("Initializing surface...")
        self.label_pr.setText('Initializing surface...')
        self.surface = trimesh.creation.icosphere(subdivisions=4, radius=self.r)
        self.surface = self.surface.apply_transform([[1, 0, 0, ci], [0, 1, 0, cj], [0, 0, 1, ck], [0, 0, 0, 1]])

        # update the surface attributes
        self.num_vertices = self.surface.vertices.shape[0]
        self.num_faces = self.surface.faces.shape[0]
        self.vertices = np.array(self.surface.vertices)
        self.faces = np.array(self.surface.faces)
        self.vertex_neighbors_idx = List([np.array(i) for i in self.surface.vertex_neighbors])
        # compute location of vertices in face array
        self.face_vertex_idxs = np.zeros((self.num_vertices, 6, 2), dtype=np.int32)
        self.progressBar.setValue(pv+4)
        for v in range(self.num_vertices):
            f, i = np.asarray(self.faces == v).nonzero()
            self.face_vertex_idxs[v, : i.shape[0], 0] = f
            self.face_vertex_idxs[v, : i.shape[0], 1] = i
            if i.shape[0] == 5:
                self.face_vertex_idxs[v, 5, 0] = -1
                self.face_vertex_idxs[v, 5, 1] = -1
        self.progressBar.setValue(pv + 7)
        self._progress += 12
        self.update_surface_attributes()
        #print("Brain extractor initialization complete!")

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_face_normals(num_faces, faces, vertices):
        """
        Compute face normals
        """
        face_normals = np.zeros((num_faces, 3))
        for i, f in enumerate(faces):
            local_v = vertices[f]
            a = local_v[1] - local_v[0]
            b = local_v[2] - local_v[0]
            face_normals[i] = np.array(
                (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            )
            #vec = face_normals[i]
            #l2n = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
            #face_normals[i] /= l2n
            face_normals[i] /= l2norm(face_normals[i])
        return face_normals

    @staticmethod
    def compute_face_angles(triangles: np.ndarray):
        """
        Compute angles in triangles of each face
        """
        # don't copy triangles
        triangles = np.asanyarray(triangles, dtype=np.float64)

        # get a unit vector for each edge of the triangle
        u = triangles[:, 1] - triangles[:, 0]
        u /= l2normarray(u)[:, np.newaxis]
        v = triangles[:, 2] - triangles[:, 0]
        v /= l2normarray(v)[:, np.newaxis]
        w = triangles[:, 2] - triangles[:, 1]
        w /= l2normarray(w)[:, np.newaxis]

        # run the cosine and per-row dot product
        result = np.zeros((len(triangles), 3), dtype=np.float64)
        # clip to make sure we don't float error past 1.0
        result[:, 0] = np.arccos(np.clip(diagonal_dot(u, v), -1, 1))
        result[:, 1] = np.arccos(np.clip(diagonal_dot(-u, w), -1, 1))
        # the third angle is just the remaining
        result[:, 2] = np.pi - result[:, 0] - result[:, 1]

        # a triangle with any zero angles is degenerate
        # so set all of the angles to zero in that case
        result[(result < 1e-8).any(axis=1), :] = 0.0
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_vertex_normals(
        num_vertices: int,
        faces: np.ndarray,
        face_normals: np.ndarray,
        face_angles: np.ndarray,
        face_vertex_idxs: np.ndarray,
    ):
        """
        Computes vertex normals

        Sums face normals connected to vertex, weighting
        by the angle the vertex makes with the face
        """
        vertex_normals = np.zeros((num_vertices, 3))
        for vertex_idx in range(num_vertices):
            face_idxs = np.asarray([f for f in face_vertex_idxs[vertex_idx, :, 0] if f != -1])
            inface_idxs = np.asarray([f for f in face_vertex_idxs[vertex_idx, :, 1] if f != -1])
            surrounding_angles = face_angles.ravel()[face_idxs * 3 + inface_idxs]
            vertex_normals[vertex_idx] = np.dot(surrounding_angles / surrounding_angles.sum(), face_normals[face_idxs])
            vertex_normals[vertex_idx] /= l2norm(vertex_normals[vertex_idx])
        return vertex_normals

    def rebuild_surface(self):
        """
        Rebuilds the surface mesh for given updated vertices
        """
        self.update_surface_attributes()
        self.surface = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    @staticmethod
    @jit(nopython=True, cache=True)
    def update_surf_attr(vertices: np.ndarray, neighbors_idx: list):
        # the neighbors array is tricky because it doesn't
        # have the structure of a nice rectangular array
        # we initialize it to be the largest size (6) then we
        # can make a check for valid vertices later with neighbors size
        neighbors = np.zeros((vertices.shape[0], 6, 3))
        neighbors_size = np.zeros(vertices.shape[0], dtype=np.int8)
        for i, ni in enumerate(neighbors_idx):
            for j, vi in enumerate(ni):
                neighbors[i, j, :] = vertices[vi]
            neighbors_size[i] = j + 1

        # compute centroids
        centroids = np.zeros((vertices.shape[0], 3))
        for i, (n, s) in enumerate(zip(neighbors, neighbors_size)):
            centroids[i, 0] = np.mean(n[:s, 0])
            centroids[i, 1] = np.mean(n[:s, 1])
            centroids[i, 2] = np.mean(n[:s, 2])

        # return optimized surface attributes
        return neighbors, neighbors_size, centroids

    def update_surface_attributes(self):
        """
        Updates attributes related to the surface
        """
        init = self._progress
        self.progressBar.setValue(init+1)
        self.triangles = self.vertices[self.faces]
        self.face_normals = self.compute_face_normals(self.num_faces, self.faces, self.vertices)
        self.progressBar.setValue(init+1)
        self.face_angles = self.compute_face_angles(self.triangles)
        self.progressBar.setValue(init+1)
        self.vertex_normals = self.compute_vertex_normals(
            self.num_vertices, self.faces, self.face_normals, self.face_angles, self.face_vertex_idxs
        )
        self.progressBar.setValue(init+1)
        self.vertex_neighbors, self.vertex_neighbors_size, self.vertex_neighbors_centroids = self.update_surf_attr(
            self.vertices, self.vertex_neighbors_idx
        )
        self.progressBar.setValue(init+1)
        self.l = self.get_mean_intervertex_distance(self.vertices, self.vertex_neighbors, self.vertex_neighbors_size)
        self.progressBar.setValue(init+1)
        self._progress += 6
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_mean_intervertex_distance(vertices: np.ndarray, neighbors: np.ndarray, sizes: np.ndarray):
        """
        Computes the mean intervertex distance across the entire surface
        """
        mivd = np.zeros(vertices.shape[0])
        for v in range(vertices.shape[0]):
            vecs = vertices[v] - neighbors[v, : sizes[v]]
            vd = np.zeros(vecs.shape[0])
            for i in range(vecs.shape[0]):
                vd[i] = l2norm(vecs[i])
            mivd[v] = np.mean(vd)
        return np.mean(mivd)

    def run(self):
        """
        Runs the extraction step.

        This deforms the surface based on the method outlined in"

        Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
        2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568;
        PMCID: PMC6871816.

        """
        iterations = self.n_iter
        #print("Running surface deformation...")
        # initialize s_vectors
        s_vectors = np.zeros(self.vertices.shape)

        # initialize s_vector normal/tangent
        s_n = np.zeros(self.vertices.shape)
        s_t = np.zeros(self.vertices.shape)

        # initialize u components
        u1 = np.zeros(self.vertices.shape)
        u2 = np.zeros(self.vertices.shape)
        u3 = np.zeros(self.vertices.shape)
        u = np.zeros(self.vertices.shape)
        pv = self._progress
        self.progressBar.setValue(pv+1)
        self._progress += 20

        # surface deformation loop
        for i in range(iterations):
            #print("Iteration: %d" % i, end="\r")
            self.label_pr.setText('{}/{}'.format(i+1, iterations))
            self.progressBar.setValue(self._progress)

            self._progress += 1
            if self._progress>100:
                self._progress = 0
            # run one step of deformation

            self.step_of_deformation(
                self.data,
                self.vertices,
                self.vertex_normals,
                self.vertex_neighbors_centroids,
                self.l,
                self.t2,
                self.t,
                self.tm,
                self.t98,
                self.E,
                self.F,
                self.bt,
                self.d1,
                self.d2,
                s_vectors,
                s_n,
                s_t,
                u1,
                u2,
                u3,
                u,
            )
            # update vertices
            self.vertices += u

            # just update the surface attributes
            self.update_surface_attributes()

        self._progress = 90
        self.progressBar.setValue(self._progress)
        # update the surface
        self.rebuild_surface()
        self.label_pr.setText('Post Processing...')

        #print("")
        #print("Complete.")






    @staticmethod
    @jit(nopython=True, cache=True)
    def step_of_deformation(
        data: np.ndarray,
        vertices: np.ndarray,
        normals: np.ndarray,
        neighbors_centroids: np.ndarray,
        l: float, # mean invertext distance
        t2: float,
        t: float,
        tm: float,
        t98: float,
        E: float,
        F: float,
        bt: float,
        d1: float,
        d2: float,
        s_vectors: np.ndarray,
        s_n: np.ndarray,
        s_t: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        u3: np.ndarray,
        u: np.ndarray,
    ):
        """
        Finds a single displacement step for the surface
        """
        for i, vertex in enumerate(vertices):
            # compute s vector
            s_vectors[i] = neighbors_centroids[i] - vertex

            # split s vector into normal and tangent components
            s_n[i] = np.dot(s_vectors[i], normals[i]) * normals[i]
            s_t[i] = s_vectors[i] - s_n[i]

            # set component u1
            u1[i] = 0.5 * s_t[i]

            # compute local radius of curvature (eq 4)
            r = (l ** 2) / (2 * l2norm(s_n[i]))

            # compute f2 # fractional update constant
            f2 = (1 + np.tanh(F * (1 / r - E))) / 2

            # set component u2
            u2[i] = f2 * s_n[i]

            # get endpoints directed interior (distance set by d1 and d2)
            e1 = closest_integer_point(vertex - d1 * normals[i])
            e2 = closest_integer_point(vertex - d2 * normals[i])

            # get lines created by e1/e2
            c = closest_integer_point(vertex)
            i1 = bresenham3d(c, e1, data.shape)
            i2 = bresenham3d(c, e2, data.shape)

            # get Imin/Imax

            linedata1 = [data[d[0], d[1], d[2]] for d in i1]
            linedata1.append(tm)
            linedata1 = np.asarray(linedata1)
            Imin = np.max(np.asarray([t2, np.min(linedata1)])) # min intensity from far distance

            linedata2 = [data[d[0], d[1], d[2]] for d in i2]

            linedata2.append(t)
            linedata2 = np.asarray(linedata2)
            # in the original paper is tm but here we use t98 which is correct
            Imax = np.min(np.asarray([t98, np.max(linedata2)])) # max intensity from close distance

            # get tl #locally appropriate intensity threshold
            tl = (Imax - t2) * bt + t2

            # compute f3
            f3 = 0.05 * 2 * (Imin - tl) / (Imax - t2) * l

            # get component u3
            u3[i] = f3 * normals[i]
        # get displacement vector
        u[:, :] = u1 + u2 + u3

    @staticmethod
    def check_bound(img_min: int, img_max: int, img_start: int, img_end: int, vol_start: int, vol_end: int):
        if img_min < img_start:
            vol_start = vol_start + (img_start - img_min)
            img_min = 0
        if img_max > img_end:
            vol_end = vol_end - (img_max - img_end)
            img_max = img_end
        return img_min, img_max, img_start, img_end, vol_start, vol_end

    def compute_mask(self):
        """
        Convert surface mesh to volume
        """
        vol = self.surface.voxelized(1)
        vol = vol.fill()
        self.mask = np.zeros(self.shape)
        bounds = vol.bounds

        # adjust bounds to handle data outside the field of view

        # get the bounds of the volumized surface mesh
        x_min = int(vol.bounds[0, 0]) if vol.bounds[0, 0] > 0 else int(vol.bounds[0, 0]) - 1
        x_max = int(vol.bounds[1, 0]) if vol.bounds[1, 0] > 0 else int(vol.bounds[1, 0]) - 1
        y_min = int(vol.bounds[0, 1]) if vol.bounds[0, 1] > 0 else int(vol.bounds[0, 1]) - 1
        y_max = int(vol.bounds[1, 1]) if vol.bounds[1, 1] > 0 else int(vol.bounds[1, 1]) - 1
        z_min = int(vol.bounds[0, 2]) if vol.bounds[0, 2] > 0 else int(vol.bounds[0, 2]) - 1
        z_max = int(vol.bounds[1, 2]) if vol.bounds[1, 2] > 0 else int(vol.bounds[1, 2]) - 1

        # get the extents of the original image
        x_start = 0
        y_start = 0
        z_start = 0
        x_end = int(self.shape[0])
        y_end = int(self.shape[1])
        z_end = int(self.shape[2])

        # get the extents of the volumized surface
        x_vol_start = 0
        y_vol_start = 0
        z_vol_start = 0
        x_vol_end = int(vol.matrix.shape[0])
        y_vol_end = int(vol.matrix.shape[1])
        z_vol_end = int(vol.matrix.shape[2])

        # if the volumized surface mesh is outside the extents of the original image
        # we need to crop this volume to fit the image
        x_min, x_max, x_start, x_end, x_vol_start, x_vol_end = self.check_bound(
            x_min, x_max, x_start, x_end, x_vol_start, x_vol_end
        )
        y_min, y_max, y_start, y_end, y_vol_start, y_vol_end = self.check_bound(
            y_min, y_max, y_start, y_end, y_vol_start, y_vol_end
        )
        z_min, z_max, z_start, z_end, z_vol_start, z_vol_end = self.check_bound(
            z_min, z_max, z_start, z_end, z_vol_start, z_vol_end
        )
        self.mask[x_min:x_max, y_min:y_max, z_min:z_max] = vol.matrix[
            x_vol_start:x_vol_end, y_vol_start:y_vol_end, z_vol_start:z_vol_end
        ]
        return self.mask



    def save_surface(self, filename: str):
        """
        Save surface in .stl
        """
        self.surface.export(filename)
    def step_of_deformation_python(self,
        data: np.ndarray,
        vertices: np.ndarray,
        normals: np.ndarray,
        neighbors_centroids: np.ndarray,
        l: float,
        t2: float,
        t: float,
        tm: float,
        t98: float,
        E: float,
        F: float,
        bt: float,
        d1: float,
        d2: float,
        s_vectors: np.ndarray,
        s_n: np.ndarray,
        s_t: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        u3: np.ndarray,
        u: np.ndarray,
    ):
        """
        Finds a single displacement step for the surface
        """
        #aaa = self.new_test(data,vertices,normals,neighbors_centroids,l,t2,t,tm,t98,E,F,bt,d1,d2,s_vectors,s_n,s_t,u1,u2,u3,u)
        # loop over vertices
        s_vectors = neighbors_centroids - vertices
        s_n = (s_vectors[:, 0] * normals[:, 0] + s_vectors[:, 1] * normals[:, 1] + s_vectors[:, 2] * normals[:, 2]).reshape(
            -1, 1) * normals
        s_t = s_vectors - s_n
        u1 = 0.5*s_t
        # compute local radius of curvature
        lnormsfdf = np.sqrt(s_n[:,0] ** 2 + s_n[:,1] ** 2 + s_n[:,2] ** 2)
        r = (l ** 2) / (2 * lnormsfdf)
        # compute f2
        f2 = (1 + np.tanh(F * (1 / r - E))) / 2
        # set component u2
        u2 = f2.reshape(-1,1) * s_n

        # closest integer point can be used instead
        # get endpoints directed interior (distance set by d1 and d2)
        e1 = np.int32(vertices - d1 * normals)
        e2 = np.int32(vertices - d2 * normals)
        # get lines created by e1/e2
        c = vertices.astype('int')
        Imin = bresenhamlines_getdata(c, e1, data, tm, t2, 'min')
        Imax = bresenhamlines_getdata(c, e2, data, t, tm, 'max')
        # get tl
        tl = (Imax - t2) * bt + t2

        # compute f3
        f3 = 0.05 * 2 * (Imin - tl) / (Imax - t2) * l

        # get component u3
        u3 = f3.reshape(-1,1) * normals
        # get displacement vector
        u[:, :] = u1 + u2 + u3
def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    file = '/home/binibica/Paper_seg/results/prep/2_X_20180202_X_t1_withoutmask.nii.gz'
    import nibabel as nib
    m = nib.load(file)
    res = m.header["pixdim"][1]
    nibf = m.get_fdata()
    from utils.utils import standardize
    #nibf = standardize(nibf)
    window = BET()
    window.setData(nibf, res)
    window.set_pars()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    run()

