from melage.utils.utils import convert_to_ras
import nibabel as nib
def changeCoordSystem(im, target):
    """
    Changing coordinate system of image
    :param target:
    :return:
    """
    previous_affine = im.affine
    previous_header = im.header
    transform, source_system = convert_to_ras(im.affine, target=target)
    if source_system == target:
        return im
    im = im.as_reoriented(transform)
    return  nib.Nifti1Image(im.get_fdata(),affine=previous_affine, header=previous_header)
