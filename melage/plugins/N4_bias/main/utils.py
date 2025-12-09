from melage.utils.utils import read_nib_as_sitk, sitk, Threshold_MultiOtsu, make_image_using_affine, nib, make_affine

def N4_bias_correction(image_nib, use_otsu=True, shrinkFactor=1,
                       numberFittingLevels=6, max_iter=5):
    inputImage = read_nib_as_sitk(image_nib)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    if use_otsu:
        #maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        threshold_val = Threshold_MultiOtsu(image_nib.get_fdata(), 1)[0]
        a = image_nib.get_fdata().copy()
        a[a <= threshold_val] = 0
        a[a > threshold_val] = 1
        mask_image = make_image_using_affine(a, image_nib.affine)
        maskImage = read_nib_as_sitk(mask_image)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)

    else:

        mask_image = nib.Nifti1Image((image_nib.get_fdata()>0).astype(np.int8), image_nib.affine, header=image_nib.header)
        #maskImage = sitk.Cast(sitk.GetImageFromArray((image.get_fdata()>0).astype('int'), sitk.sitkInt8), sitk.sitkUInt8)
        maskImage = read_nib_as_sitk(mask_image)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)


    if shrinkFactor > 1:
        inputImage = sitk.Shrink(
            inputImage, [shrinkFactor] * inputImage.GetDimension()
        )
        maskImage = sitk.Shrink(
            maskImage, [shrinkFactor] * inputImage.GetDimension()
        )


    corrector = sitk.N4BiasFieldCorrectionImageFilter()



    if max_iter > 5:
        corrector.SetMaximumNumberOfIterations(
            [max_iter] * numberFittingLevels
        )

    corrected_image = corrector.Execute(inputImage, maskImage)
    affine = make_affine(corrected_image)
    nib_im = nib.Nifti1Image(sitk.GetArrayFromImage(corrected_image).transpose(), affine)
    return nib_im