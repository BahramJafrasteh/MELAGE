import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import label as label_connector
__AUTHOR__ = 'Bahram Jafrasteh'
code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))
def convert_to_ras(affine, target = "RAS"):
    """
    Args:
        affine: affine matrix
        target: target system

    Returns:

    """
    from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform
    orig_orient = nib.io_orientation(affine)
    source_system = ''.join(list(aff2axcodes(affine, code_direction)))# get direction
    target_orient = axcodes2ornt(target, code_direction)
    transform = ornt_transform(orig_orient, target_orient)

    return transform, source_system
def LargestCC(segmentation, connectivity=3):
    """
    Get largets connected components
    """
    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    return labels, frequency


def make_affine(simpleITKImage):
    # https://niftynet.readthedocs.io/en/v0.2.1/_modules/niftynet/io/simple_itk_as_nibabel.html
    # get affine transform in LPS
    if simpleITKImage.GetDimension() == 4:
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
             for p in ((1, 0, 0, 0),
                       (0, 1, 0, 0),
                       (0, 0, 1, 0),
                       (0, 0, 0, 0))]
        c = np.array(c)
        c = c[:, :-1]
    elif simpleITKImage.GetDimension() == 3:
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
             for p in ((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (0, 0, 0))]
        c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine

def read_sitk_as_nib(sitk_im):
    return nib.Nifti1Image(sitk.GetArrayFromImage(sitk_im).transpose(),
                         make_affine(sitk_im), None)
def read_nib_as_sitk(image_nib, dtype=None):
    # From https://github.com/gift-surg/PySiTK/blob/master/pysitk/simple_itk_helper.py
    if dtype is None:
        dtype = np.float32#image_nib.header["bitpix"].dtype
    nda_nib = image_nib.get_fdata().astype(dtype)
    nda_nib_shape = nda_nib.shape
    nda = np.zeros((nda_nib_shape[2],
                    nda_nib_shape[1],
                    nda_nib_shape[0]),
                   dtype=dtype)

    # Convert to (Simple)ITK data array format, i.e. reorder to
    # z-y-x-components shape
    for i in range(0, nda_nib_shape[2]):
        for k in range(0, nda_nib_shape[0]):
            nda[i, :, k] = nda_nib[k, :, i]
    # Get SimpleITK image
    vector_image_sitk = sitk.GetImageFromArray(nda)
    # Update header from nibabel information
    # (may introduce some header inaccuracies?)
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    affine_nib = image_nib.affine.astype(np.float64)
    R_nib = affine_nib[0:-1, 0:-1]

    spacing_sitk = np.array(image_nib.header.get_zooms(), dtype=np.float64)
    spacing_sitk = spacing_sitk[0:R_nib.shape[0]]
    S_nib_inv = np.diag(1. / spacing_sitk)

    direction_sitk = R.dot(R_nib).dot(S_nib_inv).flatten()

    t_nib = affine_nib[0:-1, 3]
    origin_sitk = R.dot(t_nib)

    vector_image_sitk.SetSpacing(np.array(spacing_sitk).astype('double'))
    vector_image_sitk.SetDirection(direction_sitk)
    vector_image_sitk.SetOrigin(origin_sitk)
    return vector_image_sitk
def resample_to_size(im, new_size, scale_factor=None,method='linear'):
    """
    Resample image to new size
    """
    original_image = read_nib_as_sitk(im)
    # Get the current size of the image
    size = original_image.GetSize()

    # Calculate the scale factor for resizing
    if scale_factor is None:
        scale_factor = [(float(sz)/new_sz)*spc for sz, new_sz, spc in zip(size, new_size, original_image.GetSpacing())]

    # Resample the image using the scale factor
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetOutputSpacing(scale_factor)
    resampler.SetSize([int(el) for el in new_size])

    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetOutputDirection(original_image.GetDirection())

    if method.lower() == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)  # You can choose different interpolators
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)  # You can choose different interpolators
    # Perform resampling
    resized_image = resampler.Execute(original_image)
    #inverse_scale_factor = [s1/float(sz) for sz, s1 in zip(*[resized_image.GetSize(), size])]
    return read_sitk_as_nib(resized_image)#, inverse_scale_factor, list(size)


def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    #print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new

def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.
    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    #print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        idx = 0
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    #print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale

def normalize_mri(img):
    # taken from npp
    #https://github.com/Novestars/Neural_Pre_Processing/blob/master/nppy/models/utils.py
    src_min, scale = getscale(img, 0, 255)
    new_data = scalecrop(img, 0, 255, src_min, scale)
    return new_data

def remove_zero(f_data, value=0):
    """
    Remove non segmented areas from image
    :param f_data:
    :param value:
    :return:
    """

    xs, ys, zs = np.where(f_data > value) #find zero values
    tol = 4

    min_max = []
    for x in [xs, ys, zs]:
        minx = min(x)-tol if min(x)-tol>1 else min(x)
        maxx = max(x) + tol if max(x) + tol < f_data.shape[0]-1 else max(x)
        min_max.append([minx, maxx])
    f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, min_max

def get_back_data(im, shape_initial, pad_zero, border_value):
    """
    Get data back to its original shape
    """
    im_fill = np.ones(shape_initial) * border_value
    im_fill[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
    pad_zero[2][0]:pad_zero[2][1] + 1] = im
    return im_fill
