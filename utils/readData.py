
__AUTHOR__ = 'Bahram Jafrasteh'

import sys
sys.path.append('..')
from utils.utils import Item, Item_equal, standardize, convert_to_ras, make_affine
import SimpleITK as sitk
from datetime import datetime
import numpy as np
import nibabel as nib
import re
import os

class readData():
    """
    This is the main file to read images
    """
    def __init__(self, only_metadata = False, type='t1', target_system='IPL'):
        """
        :param only_metadata: if only metadat needs to be read
        :param type: data tyoe
        :param target_system: target system to show
        """
        self.metadata_dict = dict() #metada dictionary
        self.phi_angles = np.zeros(shape=(0,0)) # phi angle related to vol files in GE
        self.theta_angles = np.zeros(shape=(0,0)) # phi angle related to vol files in GE
        self.only_metadata = only_metadata
        self.success = False
        self.tract = None# if tractography file is available
        self.s2c = False # sagittal to coronal
        self.metadata = {}
        self.npEdge = []
        self._npSeg = None
        self.type = type
        #if type=='t1':
        #    self.target_system = target_system#'IPL'
        #elif type == 'eco':
        self.target_system = target_system#'PLI'#'SPR'#'ARI'



    def grid_date_gen(self, arr, dim_axial, dim_sagital, dim_coronal, bModeRadius):
        """
        This function related to reading GE ultrasound data (VOL not compressed)
        :param arr:
        :param dim_axial:
        :param dim_sagital:
        :param dim_coronal:
        :param bModeRadius:
        :return:
        """
        x_r = 0
        rs = np.array(
            [(self.metadata_dict['Offset_Spacing'] + i) * self.metadata_dict['Resolution'] for i in range(dim_coronal)])
        for k in range(dim_axial):
            thetas = np.repeat(self.theta_angles[k] - np.pi / 2.0, dim_coronal)
            for j in range(dim_sagital):
                phis = np.repeat(self.phi_angles[j] - np.pi / 2.0, dim_coronal)
                arr[x_r*dim_coronal:(x_r+1)*dim_coronal, 0] = rs * np.sin(phis)  # coronal
                arr[x_r*dim_coronal:(x_r+1)*dim_coronal, 1] = -(rs * np.cos(phis) - bModeRadius) * np.sin(thetas)  # sagital
                arr[x_r*dim_coronal:(x_r+1)*dim_coronal, 2] = bModeRadius * (1 - np.cos(thetas)) + rs * np.cos(phis) * np.cos(thetas)  # axial
                #arr[:,j,k, 0] = rs * np.sin(phis)  # coronal
                #arr[:,j,k, 1] = -(rs * np.cos(phis) - bModeRadius) * np.sin(thetas)  # sagital
                #arr[:,j,k, 2] = bModeRadius * (1 - np.cos(thetas)) + rs * np.cos(phis) * np.cos(thetas)  # axial
                x_r += 1
        return arr
    def read_cartesian(self, f, data_size):
        "        This function related to reading GE ultrasound data (VOL not compressed)"
        raise ValueError('No Implemented yet')



    def manuallySetIms(self, type):
        """
        Manually set image
        :param type:
        :return:
        """
        if hasattr(self, 'header'):
            hdr = self.header
        else:
            hdr = nib.Nifti1Header()
            hdr['dim'] = np.array([3, self.npImage.shape[2], self.npImage.shape[1], self.npImage.shape[0], 1, 1, 1, 1])

        if hasattr(self, 'affine'):
            if self.s2c and hasattr(self, '_imChanged_affine'):
                affine = self._imChanged_affine
            else:
                affine = self.affine
        else:
            affine = np.eye(4)
            affine[:-1, -1] = np.array(self.ImOrigin)
            np.fill_diagonal(affine[:-1, :-1], self.ImSpacing)

        data = self.npImage
        self._imChanged = nib.Nifti1Image(data, affine, hdr)
        self._imChanged.header.set_zooms(np.array(self.ImSpacing))
        transform, self.source_system = convert_to_ras(self._imChanged.affine, self.target_system)
        self._imChanged = self._imChanged.as_reoriented(transform)
        self.im = self._imChanged.__class__(self._imChanged.dataobj[:], self._imChanged.affine, self._imChanged.header)
        if hasattr(self, 'npImages'):
            self.ims = nib.Nifti1Image(self.npImages, affine, hdr)
            self.ims = self.ims.as_reoriented(transform)


    def read_non_cartesian(self, f, data_size):
        "        This function related to reading GE ultrasound data (VOL not compressed)"
        from vtk.util import numpy_support
        import vtk

        total_voxels = self.metadata_dict['Coronal_Dimension']*self.metadata_dict['Sagittal_Dimension']*self.metadata_dict['Axial_Dimension']
        #data_points_arr = np.zeros(shape=(self.metadata_dict['Coronal_Dimension'], self.metadata_dict['Sagittal_Dimension'], self.metadata_dict['Axial_Dimension'],3))
        data_points_arr = np.zeros(shape=(total_voxels,3))
        data_points = vtk.vtkPoints()
        StructuredGrid = vtk.vtkStructuredGrid()
        assert (len(self.phi_angles) == self.metadata_dict['Sagittal_Dimension'])
        assert (len(self.theta_angles) == self.metadata_dict['Axial_Dimension'])
        assert (data_size == total_voxels)
        bModeRadius = -self.metadata_dict['Offset_Radius'] * self.metadata_dict['Resolution']

        data_points_arr = self.grid_date_gen(data_points_arr,
                           self.metadata_dict['Axial_Dimension'],
                           self.metadata_dict['Sagittal_Dimension'],
                           self.metadata_dict['Coronal_Dimension'],
                           bModeRadius)
        data_points_vtk = numpy_support.numpy_to_vtk(num_array=data_points_arr, deep=True, array_type=vtk.VTK_DOUBLE)
        data_points.SetData(data_points_vtk)

        StructuredGrid.SetPoints(data_points)
        StructuredGrid.SetExtent(
            0, self.metadata_dict['Coronal_Dimension'] -1,
            0, self.metadata_dict['Sagittal_Dimension']-1,
            0, self.metadata_dict['Axial_Dimension']-1
        )

        # read bytes
        byte_voxel_values = f.read(data_size)
        # convert to voxel values
        array_voxel_values = np.asarray([int(i) for i in byte_voxel_values])
        # numpy to vtk int array
        voxel_values = numpy_support.numpy_to_vtk(num_array=array_voxel_values.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        # set voxel name
        voxel_values.SetName('VoxelIntensity')
        # assign cell data
        StructuredGrid.GetPointData().AddArray(voxel_values)
        StructuredGrid.GetPointData().SetActiveAttribute('VoxelIntensity', vtk.vtkDataSetAttributes.SCALARS)
        #StructuredGrid.GetPointData().SetScalars(voxel_values)

        # bounds
        Cartesian_Bounds = StructuredGrid.GetBounds()

        outputSpacing = [0.3, 0.3, 0.3]
        volumeDims = [int(np.ceil((Cartesian_Bounds[1] - Cartesian_Bounds[0]) / outputSpacing[0])),
                      int(np.ceil((Cartesian_Bounds[3] - Cartesian_Bounds[2]) / outputSpacing[1])),
                      int(np.ceil((Cartesian_Bounds[5] - Cartesian_Bounds[4]) / outputSpacing[2]))]
        ImageResampler = vtk.vtkResampleToImage()
        ImageResampler.SetInputDataObject(StructuredGrid)
        ImageResampler.SetSamplingDimensions(volumeDims)
        ImageResampler.Update()
        self.VtkImage = ImageResampler.GetOutput()
        if self.VtkImage.GetPointData().RemoveArray('vtkValidPointMask') is not None:
            self.VtkImage.GetPointData().RemoveArray('vtkValidPointMask')



        self.ImExtent = self.VtkImage.GetExtent()

        self.ImSpacing = self.VtkImage.GetSpacing()

        self.ImOrigin = self.VtkImage.GetOrigin()

        self.ImEnd = np.zeros(shape=(3,))
        self.ImCenter = np.zeros(shape=(3,))
        for i in range(3):
            self.ImEnd[i] = self.ImOrigin[i] + (self.ImExtent[i*2+1]-self.ImExtent[i*2])*self.ImSpacing[i]

        self.ImCenter[0] = self.ImOrigin[0] + self.ImSpacing[0] * 0.5 * (self.ImExtent[0] + self.ImExtent[1])
        self.ImCenter[1] = self.ImOrigin[1] + self.ImSpacing[1] * 0.5 * (self.ImExtent[2] + self.ImExtent[3])
        self.ImCenter[2] = self.ImOrigin[2] + self.ImSpacing[2] * 0.5 * (self.ImExtent[4] + self.ImExtent[5])
        self.metadata_dict['ImCenter'] = self.ImCenter.tolist()
        self.metadata_dict['ImEnd'] = self.ImEnd.tolist()
        self.metadata_dict['ImOrigin'] = self.ImOrigin
        self.metadata_dict['ImSpacing'] = self.ImSpacing
        self.metadata_dict['ImExtent'] = self.ImExtent

    def get_metadata(self):
        """
        Get metadat
        :return:
        """
        return self.metadata_dict



    def changeImData(self, im, axis =[2, 1, 0]):
        """
        Chamge image information
        :param im:
        :param axis:
        :return:
        """
        data = im.get_fdata()
        self.npImage = standardize(data).astype(np.uint8)



    def changeData(self, type, axis =[2, 1, 0], imchange=False, state= True, npSeg=None):
        """
        Change image data in case of changing sagittal to coronal or vice versa
        :param type:
        :param axis:
        :param imchange:
        :param state:
        :param npSeg:
        :return:
        """
        if self.target_system=='IPL' and state:
            return

        transpose_axis_inv = axis
        if state:
            transform, _ = convert_to_ras(self.im.affine, target='SRA')
            self.s2c = True
        else:
            transform, _ = convert_to_ras(self.im.affine, self.target_system)
            self.s2c = False

        im = self.im.as_reoriented(transform)
        self.ImDirection = nib.aff2axcodes(im.affine)
        self.ImExtent = (0, im.header['dim'][transpose_axis_inv[0]+1], 0,
                         im.header['dim'][transpose_axis_inv[1]+1], 0,
                         im.header['dim'][transpose_axis_inv[2]+1])
        self.ImSpacing = im.header['pixdim'][1:4][transpose_axis_inv]
        self.metadata = {key: im.header[key] for key in im.header.keys()}
        self.metadata['rot_axial'] = 0
        self.metadata['rot_sagittal'] = 0
        self.metadata['rot_coronal'] = 0

        self.ImOrigin = np.array([self.metadata['qoffset_x'].item(),
                                  self.metadata['qoffset_y'].item(),
                                  self.metadata['qoffset_z'].item()])[transpose_axis_inv] #qoffset_x, qoffset_y, qoffset_z
        self.ImEnd = np.zeros(shape=(3,))
        self.ImCenter = np.zeros(shape=(3,))
        for i in range(3):
            self.ImEnd[i] = self.ImOrigin[i] + (self.ImExtent[i * 2 + 1] - self.ImExtent[i * 2]) * self.ImSpacing[i]

        self.ImCenter[0] = self.ImOrigin[0] + self.ImSpacing[0] * 0.5 * (self.ImExtent[0] + self.ImExtent[1])
        self.ImCenter[1] = self.ImOrigin[1] + self.ImSpacing[1] * 0.5 * (self.ImExtent[2] + self.ImExtent[3])
        self.ImCenter[2] = self.ImOrigin[2] + self.ImSpacing[2] * 0.5 * (self.ImExtent[4] + self.ImExtent[5])

        if imchange:
            self._imChanged = im
            self._imChanged_affine = im.affine
            self.npImage = im.get_fdata()
            if npSeg is not None:
                self.npSeg = npSeg.as_reoriented(transform).get_fdata()
            else:
                self.npSeg = np.zeros_like(self.npImage)


    def updateData(self, im, rotm, type):
        """
        update image data
        :param im:
        :param rotm:
        :param type:
        :return:
        """
        if type == 't1':
            data = im
        elif type == 'eco':
            data = im

        self.npImage = standardize(data).astype(np.uint8)
        if type=='t1':
            self.npSeg = np.zeros_like(self.npImage).astype('int')


    def GetParams(self):
        """
        Get parameters for saving the changes
        :return:
        """
        return ['basefile']


    def readNRRD(self, file, type='eco'):
        """
        Read NRRD image data (This function should be checked)
        :param file:
        :param type:
        :return:
        """
        import nrrd
        import nibabel as nib
        _nrrd = nrrd.read(file)
        data = _nrrd[0]
        header = _nrrd[1]
        # save nifti
        if type=='eco':
            spacing = header['spacings']
            affine = np.eye(4)
            np.fill_diagonal(affine[:-1, :-1], spacing)
            self.im = nib.Nifti1Image(np.flip(np.transpose(data, [0,2,1]), 1), affine)
            transform, self.source_system = convert_to_ras(self.im.affine, target=self.target_system)
            self.im = self.im.as_reoriented(transform)
        else:
            affine = np.eye(4)
            a = np.transpose(data, [0, 2, 1])
            self.im = nib.Nifti1Image(a, affine, None)
            transform, self.source_system = convert_to_ras(self.im.affine, target='PSL')
            self.im = self.im.as_reoriented(transform)

            #raise exit('NOT IMPELEMENTED ERROR')
            #img = nib.Nifti1Image(np.flip(np.transpose(data, [0,2,1]),None), np.eye(4))

        found_image_data = True
        found_meta_data = False
        file_path, file_extension = os.path.splitext(file)
        self.basefile = os.path.basename(file)
        if os.path.isfile(file_path + '.json'):
            found_meta_data = True
        if found_image_data:
            self.success = True
        self.set_metadata()
        self.read_pars()

        return [found_meta_data, found_image_data, 'Success']

    def UpdateAnotherDim(self, dim_active=0): # just in case of 4 dimensional image
        """
        This function have been implemented to show four dimensional images suchs fmri or dwi
        :param dim_active:
        :return:
        """
        if not hasattr(self, 'ims') or dim_active<0:
            return
        if self.target_system!='IPL':
            return
        from nibabel.funcs import four_to_three
        self.im = four_to_three(self.ims)[dim_active]
        if hasattr(self, 'bvals_dwi') and hasattr(self, 'bvecs_dwi'):
            self._fileDicom = self._fileDicom_base + 'B_{}_Bvec_{}'.format(self.bvals_dwi[dim_active], np.round(self.bvecs_dwi[dim_active],2))+'.dcm'
        spacing = self.im.header['pixdim'][1:4]
        minSpacing = 1.0#np.min(spacing)
        if max(abs(np.min(spacing)-spacing))/3.0> 0.01: # check if need resampling
            self._resampling(spacing, minSpacing)
        transform, self.source_system = convert_to_ras(self.im.affine, target=self.target_system)
        self.im = self.im.as_reoriented(transform)
        found_image_data = True
        found_meta_data = False
        if found_image_data:
            self.success = True
        self.set_metadata()
        self.read_pars()
        return [found_meta_data, found_image_data, 'Success']


    def _changeCoordSystem(self, target):
        """
        Changing coordinate system of image
        :param target:
        :return:
        """
        if not hasattr(self, 'npSeg'):
            return False
        if not hasattr(self.im, 'affine'):
            return False
        affine_previous = self.im.affine.copy()
        transform, source_system = convert_to_ras(self.im.affine, target=target)
        if source_system== target:
            return False
        self.im = self.im.as_reoriented(transform)
        self.set_metadata()
        self.read_pars(reset_seg=False)
        if self.npSeg.max()>0:
            im_tm = nib.Nifti1Image(self.npSeg, affine_previous, dtype=np.int64)
            im_tm = im_tm.as_reoriented(transform)
            self.npSeg = im_tm.get_fdata()
        if not hasattr(self, 'source_system'):
            self.source_system = source_system
        return True


    def readDicomDirectory(self, file, type='econ'):
        """
        Reading a dicom directory
        :param file:
        :param type:
        :return:
        """
        from pydicom.filereader import dcmread
        from pydicom.filereader import read_dicomdir
        dicom_dir = read_dicomdir(file)
        base_dir = os.path.dirname(file)
        # go through the patient record and print information
        series_DESC_total = []
        size_total = []
        file_total = []
        for series in dicom_dir.patient_records[0].children[0].children:

            image_records = series.children

            image_filenames = [os.path.join(base_dir, *image_rec.ReferencedFileID)
                               for image_rec in image_records]
            file_reader = sitk.ImageFileReader()
            base_dir_c = os.path.dirname(image_filenames[0])
            if not os.path.isfile(image_filenames[0]):
                continue

            file_reader.SetFileName(image_filenames[0])
            file_reader.ReadImageInformation()
            try:
                series_ID, r, c, series_DESC ='', 0, 0, 'Image'
                for i, el in enumerate(['0020|000e', '0028|0010', '0028|0011', '0008|103e']):
                    if el in file_reader.GetMetaDataKeys():
                        read = file_reader.GetMetaData(el)
                        if i == 0:
                            series_ID = read
                        elif i == 1:
                            r = int(read)
                        elif i == 2:
                            c = int(read)
                        elif i == 3:
                            series_DESC = read

                sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(base_dir_c, series_ID)
                series_DESC_total.append(series_DESC)
                size_total.append([r, c, len(sorted_file_names)])
                file_total.append(sorted_file_names)

            except:
                continue

        from utils.utils import create_combo_box_new
        combo = create_combo_box_new(series_DESC_total, size_total)
        if combo.exec_() == combo.accept:
            ind_sel = combo.selectedInd
        ind_sel = combo.selectedInd
        return self.readDICOM(file_total[ind_sel][0], type)

    def _find_equal_sizeImages(self, sorted_file_names, file_reader):
        # find equal size images in the directory
        dics = dict()
        file_reader.LoadPrivateTagsOn()

        echotimes = dict()

        r = 0

        while True:

            l = sorted_file_names[r]
            r += 1

            file_reader.SetFileName(l)
            file_reader.ReadImageInformation()

            try:
                #size = (int(file_reader.GetMetaData('0028|0010')),int(file_reader.GetMetaData('0028|0011')))
                size = file_reader.GetSize()
                #print(file_reader.GetMetaData('07a1|103e'), file_reader.GetMetaData('07a5|1060'), file_reader.GetMetaData('0020|1041'))

                if size not in dics:
                    dics[size] = list()
                    echotimes[size] = []
                #if len(dics[size])>45:
                #    print('a')
                if '07a1|103e' in file_reader.GetMetaDataKeys():
                    echotimes[size].append(float(file_reader.GetMetaData('07a1|103e')))
                elif '0020|0013' in file_reader.GetMetaDataKeys():#1041, 0013
                    echotimes[size].append(float(file_reader.GetMetaData('0020|0013')))

                dics[size].append(l)
            except:
                print('')
                pass
            if r >= len(sorted_file_names):
                break

        if self._dicom_image_type=='diffusion':
            if '0008|103e' in file_reader.GetMetaDataKeys():
                series_DESC = file_reader.GetMetaData('0008|103e')
            self._fileDicom = series_DESC
            return self._read_dwi_images(dics, echotimes)

        from utils.utils import show_message_box, create_combo_box_new
        if len(dics.keys())>1:
            #show_message_box('There are {} readable files in the directory'.format(len(nib_ims)))
            series_desc_t = []
            sizes = []
            for key in dics.keys():
                fl = dics[key][0]
                file_reader.SetFileName(fl)
                file_reader.ReadImageInformation()
                if '0008|103e' in file_reader.GetMetaDataKeys():
                    series_DESC = file_reader.GetMetaData('0008|103e')
                else:
                    series_DESC = 'Image'
                series_desc_t.append(series_DESC)
                sizes.append(key)
            combo = create_combo_box_new(series_desc_t, sizes)
            if combo.exec_()==combo.accept:
                ind_sel = combo.selectedInd
            ind_sel = combo.selectedInd
        elif len(dics.keys())==1:
            ind_sel = 0
            if '0008|103e' in file_reader.GetMetaDataKeys():
                series_DESC = file_reader.GetMetaData('0008|103e')
            else:
                series_DESC = 'Image'
            series_desc_t = [series_DESC]
            sizes = [size]
        else:
            #show_message_box('The directory does not contain dicom or is not readable')
            return [False, False, 'No file']

        if ind_sel is None:
            show_message_box('The directory does not contain dicom or is not readable')
            return [False, False, 'No file']

        ind_key = sizes[ind_sel]
        time_s = echotimes[ind_key]

        imgs = dics[ind_key]
        if len(time_s)>1:
            lst = list(np.argsort(time_s))
            this_set = [imgs[l] for l in lst]
        else:
            this_set = imgs
        im = sitk.ReadImage(this_set)
        affine = make_affine(im)
        nib_im =nib.Nifti1Image(sitk.GetArrayFromImage(im).transpose(), affine)
        self._fileDicom = series_desc_t[ind_sel]+'.dcm'
        return nib_im

    def _read_dwi_images(self, dics, echotimes):
        """
        If the file is detected as dwi read it
        :param dics:
        :param echotimes:
        :return:
        """
        from utils.utils import norm_dti
        from pydicom.filereader import dcmread
        Ln = [dics[key] for key in dics.keys()]
        ech = [echotimes[key] for key in echotimes.keys()]
        ind_m = np.argmin([len(a) for a in Ln])
        ind_ma = np.argmax([len(a) for a in Ln])
        if ind_m == ind_ma and len(Ln)>1:
            ind_m = np.argmin([list(dics.keys())[1][0], list(dics.keys())[0][0]])
            ind_ma = np.argmax([list(dics.keys())[1][0], list(dics.keys())[0][0]])
        image_indices = Ln[ind_m]
        echotimes = ech[ind_ma]
        image_bvec = Ln[ind_ma]
        num_images = len(image_bvec)//len(image_indices)
        indices_bvec = np.arange(0, len(image_bvec), num_images)
        bvals = dict()
        bvecs_total = dict()
        r = 0
        images = []
        for l in image_indices:
            ds_bvec = dcmread(image_bvec[indices_bvec[r]], stop_before_pixels=True)
            this_set = image_bvec[r*num_images:(r+1)*num_images]
            if len(echotimes)>=len(image_bvec):
                lst = list(np.argsort(echotimes[r * num_images:(r + 1) * num_images]))
                this_set = [this_set[l] for l in lst]
            images.append(this_set)
            try:
                bvec = ds_bvec[0x52009230].value[0][0x002111fe].value[0][0x00211146].value
            except:
                bvec = [0, 0, 0]

            ds = dcmread(l, stop_before_pixels=True)
            ds.decode()
            bval_str = ds[0x00180024].value  # (0043,1039) for GE
            bval = int(float(re.findall(r'\d+', bval_str)[0]))  # find all numbers
            bvals[r] = bval
            size = (ds[0x00280010].value, ds[0x00280011].value)

            ####### Read BVAL
            acqu_matrix = None
            if 0x00181310 in ds:
                acqu_matrix = ds[0x00181310].value  # acquisition matrix
            phase_encoding_dir = ds[0x00181312].value  # either ROW or COLUMNs *For CANON and GE can be useful
            image_position = ds[0x00200032].value  # It is the location in mm from the origin of the RCS.#
            pixel_resolution = ds[0x00280030].value
            image_orientation_xy = ds[0x00200037].value
            # [float(i) for i in re.findall(r"\d+\.\d+", file_reader.GetMetaData('0020|0037'))]
            read_v = norm_dti(image_orientation_xy[:3])
            phase_v = norm_dti(image_orientation_xy[3:])
            slice_v = norm_dti(np.cross(read_v, phase_v))
            PatientPosition = ds[0x00185100].value  # HFS # if not HFS warning
            bvec_new = norm_dti([np.dot(read_v, bvec), np.dot(phase_v, bvec), np.dot(slice_v, bvec)])
            if bvec_new[1]!=0:
                bvec_new[1] = -bvec_new[1]
            bvecs_total[r] = bvec_new
            r += 1

        num_image = len(dics.keys())
        ast = []

        if len(Ln)==1:
            print('Guessing mosaic...')
        for img in images:
            ig = sitk.ReadImage(img)
            a = sitk.GetArrayFromImage(ig).transpose()
            if len(Ln) == 1 and acqu_matrix is not None:
                acqu_matrix = [el for el in acqu_matrix if el!=0]
                h = a.shape[0] // acqu_matrix[0]
                w = a.shape[1] // acqu_matrix[-1]
                nh = a.shape[0] // h
                nw = a.shape[1] // w
                if h * w * nh * nw <= a.shape[0] * a.shape[1] and (w>1 or h>1):
                    b = np.zeros((nh, nw, h * w))
                    r = 0
                    for i in range(h):
                        st = i * nw
                        ft = (i + 1) * nw
                        for j in range(w):
                            b[..., r] = a[j * nh:(j + 1) * nh, st:ft].squeeze()
                            r += 1
                    a = b
            if a.ndim==3:
                if a.shape[-1]!=1:
                    a = a[...,None]
            ast.append(a)

        affine = make_affine(ig)
        ast = np.block(ast)
        nib_im = nib.Nifti1Image(ast, affine)
        self.bvals_dwi = [bvals[key] for key in bvals.keys()]
        self.bvecs_dwi = np.array([bvecs_total[key] for key in bvecs_total.keys()])
        return nib_im

    def _resampling(self, spacing, minSpacing):
        """
        Resample image to desired spacing
        :param spacing:
        :param minSpacing:
        :return:
        """
        from utils.utils import resize_window
        window = resize_window()
        window.label_current_spc.setText('{:.3f},{:.3f},{:.3f}'.format(spacing[0], spacing[1], spacing[2]))
        window.label_new_spc.setValue(minSpacing)
        window.exec_()
        if window._status:
            newSpacing = window.label_new_spc.value()
            from utils.utils import resample_to_spacing
            self.im = resample_to_spacing(self.im, newSpacing)
            #print('resampling')

    def readDICOM(self, file, type='eco'):
        """
        Read DICOM image
        :param file:
        :param type:
        :return:
        """
        from pydicom.filereader import dcmread

        if file.split('/')[-1].lower()=='dicomdir':
            return self.readDicomDirectory(file, type)
        file_reader = sitk.ImageFileReader()
        reader = sitk.ImageSeriesReader()
        folder_in = os.path.dirname(file)
        series_IDs = reader.GetGDCMSeriesIDs(folder_in)

        series_file_names = reader.GetGDCMSeriesFileNames(folder_in, series_IDs[0])
        file_reader.SetFileName(series_file_names[0])
        file_reader.ReadImageInformation()
        file_reader.LoadPrivateTagsOn()
        self._Manufacturer = ''
        if '0008|0070' in file_reader.GetMetaDataKeys():
            self._Manufacturer = file_reader.GetMetaData('0008|0070')
        self._dicom_image_type = ''
        if '0008|0008' in file_reader.GetMetaDataKeys():
            tag_imt = file_reader.GetMetaData('0008|0008').split("\\")
            if len(tag_imt)>=3:
                self._dicom_image_type = tag_imt[2].lower()

        try:
            series_ID = file_reader.GetMetaData('0020|000e')
            try:
                series_DESC = file_reader.GetMetaData('0008|103e')
            except:
                series_DESC = 'Image'
            sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_in, series_ID)
        except:
            single_file_mode = True
            sorted_file_names = [file]

        a = dcmread(file)
        self.im_metadata = [[a[key].name, a[key].value] for key in a.keys() if
                      a[key].name not in ['Pixel Data', 'Overlay Data']]
        del a
        self.ims  = self._find_equal_sizeImages(sorted_file_names, file_reader)

        if self.ims.ndim==4:
            from nibabel.funcs import four_to_three
            nib_images = four_to_three(self.ims)
            if type=='eco' and len(nib_images)>1:
                index_used = [r for r, l in enumerate(self.ims.shape) if l == 3 or l == 1]
                affine = self.ims.affine
                header = self.ims.header
                image = self.ims.get_fdata()
                image = np.swapaxes(image, index_used[0], -1)
                image = image.mean(-1)
                self.ims = nib.Nifti1Image(image, affine, header=header)
                nib_images = [self.ims]
                self.im = self.ims
            else:
                self.im = nib_images[0] # select the first image
                if hasattr(self, 'bvals_dwi') and hasattr(self, 'bvecs_dwi'):
                    self._fileDicom_base = self._fileDicom
                    self._fileDicom = self._fileDicom_base+'B_{}_Bvec_{}'.format(self.bvals_dwi[0], np.round(self.bvecs_dwi[0],2))+'.dcm'
            if len(nib_images)>1:
                transform, _ = convert_to_ras(self.ims.affine, target=self.target_system)
                self.npImages = self.ims.as_reoriented(transform).get_fdata()
            else:
                delattr(self, 'ims')
        else:
            self.im = self.ims
            delattr(self, 'ims')

        spacing = self.im.header['pixdim'][1:4]
        minSpacing = 1.0#np.min(spacing)
        self.set_metadata()

        if max(abs(np.min(spacing)-spacing))/3.0> 0.01: # check if need resampling
            self._resampling(spacing, minSpacing)
        self.im_metadata.append(['Affine', self.im.affine])
        transform, self.source_system = convert_to_ras(self.im.affine, target=self.target_system)
        self.im = self.im.as_reoriented(transform)
        found_image_data = True
        found_meta_data = False
        file_path, file_extension = os.path.splitext(file)
        self.basefile = os.path.basename(file)
        if os.path.isfile(file_path + '.json'):
            found_meta_data = True
        if found_image_data:
            self.success = True
        self.set_metadata()
        self.read_pars()

        return [found_meta_data, found_image_data, 'Success']

    def set_metadata(self):
        """
        Set meta data for the file being read
        :return:
        """
        self.metadata = {key: self.im.header[key] for key in self.im.header.keys()}
        self.metadata['rot_axial'] = 0
        self.metadata['rot_sagittal'] = 0
        self.metadata['rot_coronal'] = 0

    def readNIFTI(self, file, type = 'eco'):
        """
        Read file with nifti format
        :param file:
        :param type:
        :return:
        """
        ims = nib.load(file) # read image

        dtype = ims.get_data_dtype()
        if len(dtype)>0:
            print('structured array :{}'.format(dtype))
            imsg = ims.get_data()
            imsg = imsg.view((imsg.dtype[0], len(imsg.dtype.names)))
            ims = nib.Nifti1Image(imsg, ims.affine, ims.header)

        self.ims = ims
        self.im_metadata = [[key,self.ims.header[key]] for key in self.ims.header.keys()]
        self.im_metadata.append(['Affine', self.ims.affine])
        if self.ims.ndim==4:
            from nibabel.funcs import four_to_three
            nib_images = four_to_three(self.ims)
            self.im = nib_images[0] # select the first image
            if len(nib_images)>1:
                transform, _ = convert_to_ras(self.ims.affine, target=self.target_system)
                self.npImages = self.ims.as_reoriented(transform).get_fdata()
            else:
                delattr(self, 'ims')
        else:
            self.im = self.ims
            delattr(self, 'ims')
        spacing = self.im.header['pixdim'][1:4]
        minSpacing = 1.0#np.min(spacing)
        self.set_metadata()

        if max(abs(np.min(spacing)-spacing))/3.0> 0.01: # check if need resampling
            self._resampling(spacing, minSpacing)

        transform, self.source_system = convert_to_ras(self.im.affine, target=self.target_system)
        self.im = self.im.as_reoriented(transform)

        found_image_data = True
        found_meta_data = False
        file_path, file_extension = os.path.splitext(file)
        self.basefile = os.path.basename(file)
        if os.path.isfile(file_path + '.json'):
            found_meta_data = True
        if found_image_data:
            self.success = True
        self.read_pars()
        return [found_meta_data, found_image_data, 'Success']

    def read_pars(self, reset_seg=True):
        """

        :param reset_seg: if True segmentation is removed
        :return:
        """
        # assing image parameters
        self._imChanged = self.im.__class__(self.im.dataobj[:], self.im.affine, self.im.header)

        #self.affine, shape = get_affine_shape(self.im)
        self.affine = self.im.affine
        self.header = self.im.header

        data = self.im.get_fdata()
        self.npImage = standardize(data).astype(np.uint8)
        if reset_seg:
            self.npSeg = np.zeros_like(self.npImage).astype('int')
        #self.npEdge = np.empty((0,3))

        self.ImDirection = nib.aff2axcodes(self.im.affine)
        transpose_axis_inv = [2,1,0]#self.transpose_axis[::-1]
        self.ImExtent = (0, self.im.header['dim'][transpose_axis_inv[0]+1], 0,
                         self.im.header['dim'][transpose_axis_inv[1]+1], 0,
                         self.im.header['dim'][transpose_axis_inv[2]+1])

        self.ImSpacing = self.im.header['pixdim'][1:4][transpose_axis_inv]
        self.ImOrigin = np.array([self.metadata['qoffset_x'].item(),
                                  self.metadata['qoffset_y'].item(),
                                  self.metadata['qoffset_z'].item()])[transpose_axis_inv] #qoffset_x, qoffset_y, qoffset_z
        self.ImEnd = np.zeros(shape=(3,))
        self.ImCenter = np.zeros(shape=(3,))
        for i in range(3):
            self.ImEnd[i] = self.ImOrigin[i] + (self.ImExtent[i * 2 + 1] - self.ImExtent[i * 2]) * self.ImSpacing[i]

        self.ImCenter[0] = self.ImOrigin[0] + self.ImSpacing[0] * 0.5 * (self.ImExtent[0] + self.ImExtent[1])
        self.ImCenter[1] = self.ImOrigin[1] + self.ImSpacing[1] * 0.5 * (self.ImExtent[2] + self.ImExtent[3])
        self.ImCenter[2] = self.ImOrigin[2] + self.ImSpacing[2] * 0.5 * (self.ImExtent[4] + self.ImExtent[5])



    def readKretz(self, file):
        """
        Read Kretz GE healthcare
        :param file:
        :return:
        """
        import struct

        kretz_identifier = b"KRETZFILE 1.0   "
        found_image_data = False
        found_meta_data = False
        if file == '':
            return [found_meta_data, found_image_data, "No file"]
        with open(file, 'rb') as f:
            byte = f.read(16)
            if byte != kretz_identifier:
                return [found_meta_data, found_image_data, "Not a kretz file"]
            while byte:
                Item.tagcl = f.read(2)
                if not Item.tagcl:
                    break
                    # label_names = ["Patient Name", "Patient ID", "Height","Width","Channels","Resoloution",
                    #              "Time", "Hospital","Device","TIB","Technology","MI","TIS","Ultrasound Machine"]
                Item.tagel = f.read(2)
                Item.size = f.read(4)
                data_size = struct.unpack('I', Item.size)[0] # get data size
                if Item_equal(Item, (0xC000, 0x0001)): # read coronal dimension
                    found_meta_data = True
                    dim_coronal = struct.unpack('H', f.read(data_size))[0]
                    self.metadata_dict['Coronal_Dimension'] = dim_coronal

                elif Item_equal(Item, (0xC000, 0x0002)):# read sagital dimension
                    found_meta_data = True
                    dim_sagital = struct.unpack('H', f.read(data_size))[0]
                    self.metadata_dict['Sagittal_Dimension'] = dim_sagital
                elif Item_equal(Item, (0xC000, 0x0003)):# read axial dimension
                    found_meta_data = True
                    dim_axial = struct.unpack('H', f.read(data_size))[0]
                    self.metadata_dict['Axial_Dimension'] = dim_axial
                elif Item_equal(Item, (0xC100, 0x0001)):# read resolution
                    found_meta_data = True
                    resolution = struct.unpack('d', f.read(data_size))[0]
                    self.metadata_dict['Resolution'] = resolution*1000
                elif Item_equal(Item, (0xC300, 0x0002)):  # read phi angles
                    assert (int(data_size / 8) == dim_sagital)
                    data_in = f.read(data_size)
                    if data_size%8 != 0:
                        return [found_meta_data, found_image_data, "Phi angles are not readable"]
                    self.phi_angles = np.resize(self.phi_angles, (dim_sagital,))
                    for i in range(dim_sagital):
                        self.phi_angles[i] = struct.unpack('d', data_in[i * 8:(i + 1) * 8])[0]
                elif Item_equal(Item, (0xC200, 0x0001)): # offset spacing
                    found_meta_data = True
                    offset_spacing = struct.unpack('d', f.read(data_size))[0]
                    self.metadata_dict['Offset_Spacing'] = offset_spacing
                elif Item_equal(Item, (0xC200, 0x0002)):# offset radius
                    found_meta_data = True
                    offset_radius = struct.unpack('d', f.read(data_size))[0]
                    self.metadata_dict['Offset_Radius'] = offset_radius
                elif Item_equal(Item, (0xC300, 0x0001)): # theta angles should be 215
                    assert (int(data_size/8)==dim_axial)
                    data_in = f.read(data_size)
                    if data_size%8 != 0:
                        return [found_meta_data, found_image_data, "Theta angles are not readable"]
                    self.theta_angles = np.resize(self.theta_angles, (dim_axial,))
                    for i in range(dim_axial):
                        self.theta_angles[i] = struct.unpack('d', data_in[i * 8:(i + 1) * 8])[0]
                elif Item_equal(Item, (0x0010, 0x0022)): # cartesian spacing
                    found_meta_data = True
                    CartesianSpacing = struct.unpack('d', f.read(data_size))[0]
                    self.metadata_dict['Cartesian_Spacing'] = CartesianSpacing
                elif Item_equal(Item, (0x0110, 0x0001)):  # patient id
                    found_meta_data = True
                    P_ID = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Patient_ID'] = P_ID
                elif Item_equal(Item,(0x0110, 0x0002)):  # patient name
                    found_meta_data = True
                    P_name = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Patient_Name'] = P_name
                elif Item_equal(Item,(0x0140, 0x0003)):  # study date
                    found_meta_data = True
                    study_date = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    study_date = datetime.strptime(study_date, '%Y%m%d').strftime('%d/%b/%Y')
                    self.metadata_dict['Study_Date'] = study_date
                elif Item_equal(Item,(0x0140, 0x0004)):  # study time
                    found_meta_data = True
                    study_time = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    study_time = datetime.strptime(study_time, '%H%M%S').strftime('%I:%M%p')
                    self.metadata_dict['Study_Time'] = study_time
                elif Item_equal(Item,(0x0110, 0x0003)):  # birth date
                    found_meta_data = True
                    birth_date = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    birth_date = datetime.strptime(birth_date, '%Y%m%d').strftime('%d/%b/%Y')
                    self.metadata_dict['Birth_Date'] = birth_date
                elif Item_equal(Item,(0x0120, 0x0001)):  # Hospital name
                    found_meta_data = True
                    hospital_name = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Hospital_Name'] = hospital_name
                elif Item_equal(Item, (0x0130, 0x0001)): # device
                    found_meta_data = True
                    device_name = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Ultrasound_Device'] = device_name
                elif Item_equal(Item, (0x0140, 0x0002)): # tech
                    found_meta_data = True
                    tech_name = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Technology'] = tech_name
                elif Item_equal(Item, (0x0150, 0x0013)): # velocity
                    found_meta_data = True
                    velocity = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Velocity'] = velocity
                elif Item_equal(Item, (0x0150, 0x0014)): # velocity
                    found_meta_data = True
                    velocity = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Velocity2'] = velocity
                elif Item_equal(Item, (0x0150, 0x0018)): # length
                    found_meta_data = True
                    length = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['Length'] = length
                elif Item_equal(Item, (0x150, 0x29)): # Ultra sound machine
                    found_meta_data = True
                    UltrasoundMachine = f.read(data_size).decode('iso-8859-1').replace('\x00', '').replace('\x99', '')
                    self.metadata_dict['Ultrasound_Machine'] = UltrasoundMachine
                elif Item_equal(Item, (0x0150, 0x002A)): # TIs
                    found_meta_data = True
                    TIs = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    if len(TIs.split('  '))<2:
                        self.metadata_dict['TIs'] = TIs
                    else:
                        self.metadata_dict['TIs'] = float(TIs.split('  ')[1])
                elif Item_equal(Item, (0x0150, 0x002B)):  # MI
                    found_meta_data = True
                    MI = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    self.metadata_dict['MI'] = float(MI.split('  ')[1])
                elif Item_equal(Item, (0x0150, 0x0038)):  # thermal index for bone
                    found_meta_data = True
                    TIB = f.read(data_size).decode('iso-8859-1').replace('\x00', '')
                    if len(TIB.split('  '))<2:
                        self.metadata_dict['TIb'] = TIB
                    else:
                        self.metadata_dict['TIb'] = float(TIB.split('  ')[1])
                elif Item_equal(Item, (0xD000, 0x0001)): # image data
                    found_image_data = True
                    if not self.only_metadata:
                        if len(self.phi_angles)==0 or len(self.theta_angles)==0:
                            self.read_cartesian(f, data_size)
                        else:
                            self.read_non_cartesian(f, data_size)
                else:
                    d = f.read(data_size)
                    debug = False
                    if debug:
                        print(Item.tagcl)
                        print(Item.tagel)
                        if data_size == 2:
                            print(struct.unpack('H', d)[0])
                        elif data_size == 1:
                            print(struct.unpack('b', d)[0])
                        elif data_size == 4:
                            print(struct.unpack('I', d)[0])
                        elif data_size%8==0:
                            n = int(data_size/8)
                            print([struct.unpack('d', d[i * 8:(i + 1) * 8])[0] for i in range(n)])
                        else:
                            print(d)
        if found_meta_data and found_image_data:
            self.success = True
            import vtk
            nifit_writer = vtk.vtkNIFTIImageWriter()
            nifit_writer.SetInputData(self.VtkImage)
            new_file = '/'.join(file.split('.')[:-1])+'.nii.gz'
            nifit_writer.SetFileName(new_file)
            nifit_writer.Write()
            return self.readNIFTI(new_file, type=type)
            #self.npEdge = np.empty((0, 3))
        return [False, False, 'Failed']

if __name__ == "__main__":
    import os
    cd = 'folder'
    a = readData()
    a.readNIFTI(os.path.join(cd, 'test.nii.gz'), 'mri')
