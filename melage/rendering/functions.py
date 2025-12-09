
from __future__ import division

import numpy as np
from PyQt5 import QtGui

"""
taken from pyqtgraph https://github.com/pyqtgraph/pyqtgraph
"""
    



def transformToArray(tr):
    """
    Given a QTransform, return a 3x3 numpy array.
    Given a QMatrix4x4, return a 4x4 numpy array.

    Example: map an array of x,y coordinates through a transform::

        ## coordinates to map are (1,5), (2,6), (3,7), and (4,8)
        coords = np.array([[1,2,3,4], [5,6,7,8], [1,1,1,1]])  # the extra '1' coordinate is needed for translation to work

        ## Make an example transform
        tr = QtGui.QTransform()
        tr.translate(3,4)
        tr.scale(2, 0.1)

        ## convert to array
        m = pg.transformToArray()[:2]  # ignore the perspective portion of the transformation

        ## map coordinates through transform
        mapped = np.dot(m, coords)
    """
    #return np.array([[tr.m11(), tr.m12(), tr.m13()],[tr.m21(), tr.m22(), tr.m23()],[tr.m31(), tr.m32(), tr.m33()]])
    ## The order of elements given by the method names m11..m33 is misleading--
    ## It is most common for x,y translation to occupy the positions 1,3 and 2,3 in
    ## a transformation matrix. However, with QTransform these values appear at m31 and m32.
    ## So the correct interpretation is transposed:
    if isinstance(tr, QtGui.QTransform):
        return np.array([[tr.m11(), tr.m21(), tr.m31()], [tr.m12(), tr.m22(), tr.m32()], [tr.m13(), tr.m23(), tr.m33()]])
    elif isinstance(tr, QtGui.QMatrix4x4):
        return np.array(tr.copyDataTo()).reshape(4,4)
    else:
        raise Exception("Transform argument must be either QTransform or QMatrix4x4.")

def transformCoordinates(tr, coords, transpose=False):
    """
    Map a set of 2D or 3D coordinates through a QTransform or QMatrix4x4.
    The shape of coords must be (2,...) or (3,...)
    The mapping will _ignore_ any perspective transformations.
    
    For coordinate arrays with ndim=2, this is basically equivalent to matrix multiplication.
    Most arrays, however, prefer to put the coordinate axis at the end (eg. shape=(...,3)). To 
    allow this, use transpose=True.
    
    """
    
    if transpose:
        ## move last axis to beginning. This transposition will be reversed before returning the mapped coordinates.
        coords = coords.transpose((coords.ndim-1,) + tuple(range(0,coords.ndim-1)))
    
    nd = coords.shape[0]
    if isinstance(tr, np.ndarray):
        m = tr
    else:
        m = transformToArray(tr)
        m = m[:m.shape[0]-1]  # remove perspective
    
    ## If coords are 3D and tr is 2D, assume no change for Z axis
    if m.shape == (2,3) and nd == 3:
        m2 = np.zeros((3,4))
        m2[:2, :2] = m[:2,:2]
        m2[:2, 3] = m[:2,2]
        m2[2,2] = 1
        m = m2
    
    ## if coords are 2D and tr is 3D, ignore Z axis
    if m.shape == (3,4) and nd == 2:
        m2 = np.empty((2,3))
        m2[:,:2] = m[:2,:2]
        m2[:,2] = m[:2,3]
        m = m2
    
    ## reshape tr and coords to prepare for multiplication
    m = m.reshape(m.shape + (1,)*(coords.ndim-1))
    coords = coords[np.newaxis, ...]
    
    # separate scale/rotate and translation    
    translate = m[:,-1]  
    m = m[:, :-1]
    
    ## map coordinates and return
    mapped = (m*coords).sum(axis=1)  ## apply scale/rotate
    mapped += translate
    
    if transpose:
        ## move first axis to end.
        mapped = mapped.transpose(tuple(range(1,mapped.ndim)) + (0,))
    return mapped
    
