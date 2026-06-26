from melage.utils.utils import convert_to_ras
import nibabel as nib
def changeCoordSystem(im, target):
    """
    Changing coordinate system of image
    :param target:
    :return:
    """
    transform, source_system = convert_to_ras(im.affine, target=target)
    if source_system == target:
        return im
    return im.as_reoriented(transform)
