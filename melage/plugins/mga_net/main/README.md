# MGA-Net

![MGA-Net Architecture](https://github.com/BahramJafrasteh/MGA-Net/blob/main/figures/Network_Architecture.png)

## Dependencies
This software depends on the following libraries:
```
torch
numpy
nibabel
SimpleITK
Sickit-image
Scipy
```
## running the model
```
python test_mgaNet.py image.nii.gz (-1 or 1) (float value -3 to 3)
```
where image.nii.gz is an input nifti image. (-1 or 1) is a variable to define image modality (MRI=1, US=-1) and (float value -3 to 3) is the threshold to select the mask.
    
# Mask-Guided Attention U-Net for Enhanced Neonatal Brain Extraction and Image Preprocessing

In this study, we introduce MGA-Net, a novel mask-guided attention neural network, which extends the U-net model for precision neonatal brain imaging. MGA-Net is designed to extract the brain from other structures and reconstruct high-quality brain images. The network employs a common encoder and two decoders: one for brain mask extraction and the other for brain region reconstruction.

A key feature of MGA-Net is its high-level mask-guided attention module, which leverages features from the brain mask decoder to enhance image reconstruction. To enable the same encoder and decoder to process both MRI and ultrasound (US) images, MGA-Net integrates sinusoidal positional encoding. This encoding assigns distinct positional values to MRI and US images, allowing the model to effectively learn from both modalities. Consequently, features learned from a single modality can aid in learning a modality with less available data, such as US.

We extensively validated the proposed MGA-Net on diverse datasets from varied clinical settings and neonatal age groups. The metrics used for assessment included the DICE similarity coefficient, recall, and accuracy for image segmentation; structural similarity for image reconstruction; and root mean squared error for total brain volume estimation from 3D ultrasound images. Our results demonstrate that MGA-Net significantly outperforms traditional methods, offering superior performance in brain extraction and segmentation while achieving high precision in image reconstruction and volumetric analysis.


Thus, MGA-Net represents a robust and effective preprocessing tool for MRI and 3D ultrasound images, marking a significant advance in neuroimaging that enhances both research and clinical diagnostics in the neonatal period and beyond.

For more details, refer to the full paper: [Mask-Guided Attention U-Net for Enhanced Neonatal Brain Extraction and Image Preprocessing](https://arxiv.org/abs/2406.17709)
Paper accepted for publication in "NeuroImage".
### Citation
```apache
@misc{jafrasteh2024maskguidedattentionunetenhanced,
      title={Mask-Guided Attention U-Net for Enhanced Neonatal Brain Extraction and Image Preprocessing}, 
      author={Bahram Jafrasteh and Simon Pedro Lubian-Lopez and Emiliano Trimarco and Macarena Roman Ruiz and Carmen Rodriguez Barrios and Yolanda Marin Almagro and Isabel Benavente-Fernandez},
      year={2024},
      eprint={2406.17709},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2406.17709}, 
}
