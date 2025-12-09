import numpy as np
from skimage.measure import label as label_connector
import nibabel as nib
import SimpleITK as sitk

def createPathData_IXI():
    return
def creaePathData_LDM_fullhead():
    return

code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))

def read_atlas_file(atlas_file, printing=False, save_again=False):
    im = nib.load(atlas_file)
    if im.ndim == 4 and im.shape[-1]==1:
        im = nib.funcs.four_to_three(im)[0]
    orig_orient = nib.io_orientation(im.affine)
    code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    source_system = ''.join(list(aff2axcodes(im.affine, code_direction)))
    if source_system != 'RAS':
        print(f'converted to RAS {atlas_file}')
        target_orient = axcodes2ornt('RAS', code_direction)
        transform = ornt_transform(orig_orient, target_orient)
        im = im.as_reoriented(transform)
        if save_again:
            im.to_filename(atlas_file)

    if printing:
        print('{} with range {:.2f}, {:.2f}'.format(atlas_file, im.get_fdata().min(), im.get_fdata().max()))
    #atlas_ims[..., r] = im.get_fdata().copy()
    return im


def read_sitk_as_nib(sitk_im):
    return nib.Nifti1Image(sitk.GetArrayFromImage(sitk_im).transpose(),
                         make_affine(sitk_im), None)


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

def resample_to_size(im, new_size=None, method='linear',scale_factor=None):
    original_image = read_nib_as_sitk(im)
    # Get the current size of the image
    size = original_image.GetSize()
    spacing = original_image.GetSpacing()
    # Calculate the scale factor for resizing
    if scale_factor is not None:
        # Compute new size if scale_factor is provided
        new_size = [int(round(orig_sz * sf)) for orig_sz, sf in zip(size, scale_factor)]
        scale_factor = [orig_sp / sf for orig_sp, sf in zip(spacing, scale_factor)]
    else:
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
    elif method.lower() == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)  # You can choose different interpolators

    # Perform resampling
    resized_image = resampler.Execute(original_image)
    #inverse_scale_factor = [s1/float(sz) for sz, s1 in zip(*[resized_image.GetSize(), size])]
    return read_sitk_as_nib(resized_image)#, inverse_scale_factor, list(size)

def remove_zero(f_data, value=0, min_max=None):
    """
    Remove non segmented areas from image
    :param f_data:
    :param value:
    :return:
    """


    shape_original = f_data.shape
    if min_max is None:
        min_max = []
        xs, ys, zs = np.where(f_data > value)  # find zero values
        tol = 4
        for x in [xs, ys, zs]:
            minx = min(x)-tol if min(x)-tol>1 else min(x)
            maxx = max(x) + tol if max(x) + tol < f_data.shape[0]-1 else max(x)
            min_max.append([minx, maxx])
    f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, min_max, shape_original


def restore_zero(cropped_data, original_shape, min_max, value=0):
    """
    Restore the cropped data back to the original shape with padding (default = 0).
    :param cropped_data: The cropped array from remove_zero
    :param original_shape: The shape of the original full image
    :param min_max: The min/max indices used for cropping
    :param value: The padding value to fill
    :return: Restored full-size array
    """
    restored = np.full(original_shape, value, dtype=cropped_data.dtype)

    restored[
        min_max[0][0]:min_max[0][1] + 1,
        min_max[1][0]:min_max[1][1] + 1,
        min_max[2][0]:min_max[2][1] + 1
    ] = cropped_data

    return restored

def pad_to_multiple_of_32(img, target_size = [224, 256, 192], val_const=0):
    #target_size = [224, 256, 192]  # Nearest multiples of 32
     # Nearest multiples of 32
    diff = [t - s for t, s in zip(target_size, img.shape)]

    # Distribute padding symmetrically
    pad_width = [(diff[0] // 2, diff[0] - diff[0] // 2),  # Depth
                 (diff[1] // 2, diff[1] - diff[1] // 2),  # Height
                 (diff[2] // 2, diff[2] - diff[2] // 2)]  # Width

    return np.pad(img, pad_width, mode='constant', constant_values=val_const), pad_width


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

def normalize(img, mask=None):
    if mask is None:
        src_min, scale = getscale(img, 0, 255)
        new_data = scalecrop(img, 0, 255, src_min, scale)

    else:
        src_min, scale = getscale(img*mask, 0, 255)
        new_data = np.zeros_like(img)
        new_data[mask] = scalecrop(img[mask], 0, 255, src_min, scale)
        new_data[~mask] = scalecrop(img[~mask], 0, 255, src_min, scale)
    new_data = new_data / 255


    return new_data

def LargestCC(segmentation, connectivity=3):
    """
    Get largest connected components
    """
    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    return labels, frequency
