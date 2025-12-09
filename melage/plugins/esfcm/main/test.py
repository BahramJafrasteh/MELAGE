from .FCM import esFCM, FCM, FCM_pure
from .utils import remove_zero
import numpy as np
def get_inference(model_name, progressBar, im, affine, num_tissues, post_correction, max_iter):
    if affine is None:
        affine = np.eye(4) # put identity if affine is None
    use_ssim = True
    padding = 0
    image_range = 1000
    tissue_labels = None
    fuzziness = 3
    constraint = False
    InitMethod = 'otsu'
    if num_tissues > 3 and not constraint:
        post_correction = False
        InitMethod = "kmeans"

    epsilon = 5e-3
    shape_init = im.shape
    image_used, pad_zero = remove_zero(im, 0)
    if model_name.lower() == 'esfcm':

        model = esFCM(image_used, affine,
                      image_range, num_tissues, fuzziness,
                      epsilon=epsilon, max_iter=max_iter,
                      padding=padding,
                      tissuelabels=tissue_labels,
                      mask=image_used > 0, use_ssim=use_ssim)
        try:
            model.initialize_fcm(initialization_method=InitMethod)
        except:
            model.initialize_fcm(initialization_method='kmeans')

        model.fit(progressBar)
        seg1 = model.predict(use_softmax=True).astype('int')
        seg_init2 = np.zeros(shape_init)
        seg_init2[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
        pad_zero[2][0]:pad_zero[2][1] + 1] = seg1
        return seg_init2

    elif model_name.lower() == 'fcm':
        from melage.core.Segmentation.FCM import FCM_pure as FCM

        model = FCM(image_used, affine, None,
                    image_range, num_tissues, fuzziness,
                    epsilon=epsilon, max_iter=max_iter,
                    padding=padding, constraint=False, post_correction=post_correction, mask=image_used > 0)
        try:
            model.initialize_fcm(initialization_method=InitMethod)
        except:
            model.initialize_fcm(initialization_method='kmeans')
        model.fit(progressBar)
        seg1 = model.predict(use_softmax=True).astype('int')
        seg_init2 = np.zeros(shape_init)
        seg_init2[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
        pad_zero[2][0]:pad_zero[2][1] + 1] = seg1
        return seg_init2
