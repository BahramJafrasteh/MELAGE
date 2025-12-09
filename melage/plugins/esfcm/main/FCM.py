import numpy as np

from .utils import BiasCorrection, create_window_3D, ssim3D, LargestCC, \
    update_according_to_neighbours_conv, neighborhood_conv, axis_based_convolution, adjust_common_structures, \
    compute_sdf, rescale_between_a_b
from sklearn.preprocessing import PolynomialFeatures
from scipy.ndimage import binary_fill_holes, binary_dilation
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
import abc




class FCM(object):

    def __init__(self, parent=None):
        self.parent = parent


    @abc.abstractmethod
    def Update_memership(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def Update_centers(self):
        raise NotImplementedError("Subclass should implement this.")

    def predict(self, use_softmax=False):
        raise NotImplementedError("Subclass should implement this.")


    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError("Subclass should implement this.")

    def _normalizeAtlasCreateMask(self):
        raise NotImplementedError("Subclass should implement this.")

    def initialize_fcm(self, initialization_method='otsu'):

        if initialization_method=='random':
            self.rng = np.random.default_rng(0)
            m, n, o = self.image.shape
            self.Membership = self.rng.uniform(size=(m, n, o, self.num_tissues))
            self.Membership = self.Membership / np.tile(self.Membership.sum(axis=-1)[...,np.newaxis], self.num_tissues)

            self.Membership[~self.mask, :] = 0
            mask = self.mask.copy()

            numerator = np.einsum('il->l',
                                  np.expand_dims(self.image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                         self.fuzziness))

            denominator = np.einsum('il->l', pow(self.Membership[mask], self.fuzziness))
            ind_non_denom = denominator != 0
            numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]
            numerator[numerator == 0] = 0.00001
            self.Centers = numerator
        elif initialization_method=='otsu':
            from skimage.filters import threshold_multiotsu, threshold_otsu
            self.Centers = list(threshold_multiotsu(self.image[self.mask], classes=self.num_tissues+1))
            el = -2. / (self.fuzziness - 1)

            numerator = np.zeros((*self.image.shape, self.num_tissues))

            for i in range(self.num_tissues):
                numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i])+1e-7, el)
                #numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i])+1e-7, el)

            sumn = numerator.sum(-1)
            ind_non_zero = sumn != 0
            sumn = np.expand_dims(sumn, -1)
            numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            self.Membership = numerator
            self.Membership_freeB = numerator.copy()

        elif initialization_method == 'kmeans':
            from sklearn.cluster import KMeans
            km = KMeans(self.num_tissues, random_state=0).fit(self.image[self.image > 0].reshape(-1, 1))
            self.Centers = km.cluster_centers_.squeeze()
            #idx = np.arange(self.num_gray)
            #c_mesh, idx_mesh = np.meshgrid(self.Centers, idx)
            el = -2. / (self.fuzziness - 1)

            numerator = np.zeros((*self.image.shape, self.num_tissues))


            for i in range(self.num_tissues):
                numerator[self.mask>0, i] = np.power(abs(self.image[self.mask>0] - self.Centers[i]),el)

            sumn = numerator.sum(-1)
            ind_non_zero = sumn != 0
            sumn = np.expand_dims(sumn, -1)
            numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            self.Membership = numerator





class esFCM(FCM):

    def __init__(self, image, affine,
                 image_range, num_tissues, fuzziness,
                  epsilon, max_iter,
                 padding=0,
                 tissuelabels=None, correct_bias=True,
                 mask=None, max_fail=4, use_ssim=True):
        super(esFCM, self).__init__()
        self.biascorrection = BiasCorrection()
        self.use_ssim = use_ssim
        self.mask = mask
        self.image = image
        self._imdim = [sh == 1 for sh in self.image.shape]
        self._is2D = sum([sh == 1 for sh in self.image.shape])>0
        self.max_fail = max_fail
        self.window = create_window_3D(11, 1)

        self.estimate = image.copy()  # wstep
        self.weight = image.copy()  # wstep
        self.type_im = 'T1'
        self.image_range = image_range
        self.num_tissues = num_tissues
        self.fuzziness = fuzziness
        self.padding = padding
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.correct_bias = correct_bias

        self.shape = image.shape  # image shape
        # flatted image shape: (number of pixels,1)

        self.affine = affine
        self.biasfield = 1

        if tissuelabels is not None:
            self.tissuelabels = tissuelabels.astype('int')
        else:
            #
            if self.num_tissues == 3:
                self.tissuelabels = np.array([2, 3, 4])
            else:
                self.tissuelabels = np.zeros(self.num_tissues)

    def SetBiasField(self, biasfield):
        self.biasfield = biasfield

    def Update_membership(self, constraint=True):
        """
        Updating FCM membership function
        @param final:
        @param membership_p:
        @return:
        """

        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        for i in range(self.num_tissues):
            weight = 1
            numerator[self.mask, i] = weight * np.power(abs(self.filtered_image[self.mask] - self.Centers[i]) + 1e-7,
                                                        el)

        sumn = numerator.sum(-1)
        numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])

        ind_non_zero_maks = self.mask == 1

        mrf_energy = self._proximity_measure(ind_non_zero_maks, self.Membership)

        numerator *= mrf_energy




        return numerator


    def Update_centers(self):
        """
        Update center of the clusters
        @return:
        """
        mask = self.mask.copy()

        numerator = np.einsum('il->l',
                              np.expand_dims(self.filtered_image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                     self.fuzziness))
        denominator = np.einsum('il->l', pow(self.Membership[mask], self.fuzziness))
        ind_non_denom = denominator != 0
        numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]

        numerator[numerator == 0] = 0.00001  # for numerical stability
        return numerator

    def WStep(self):
        """
        WStep
        @return:
        """

        input_inp = self.image

        a = self.predict()
        uq = np.unique(a)
        uq = [u for u in uq if u != 0]
        pred = np.zeros((*a.shape, self.num_tissues))
        for i, u in enumerate(uq):
            ind = a == u
            pred[ind, i] = 1

        numstd = np.einsum('ijkl,ijkl->ijk', self.Membership, (input_inp[..., None] - self.Centers) ** 2)
        denominator = np.sqrt(numstd)

        numerator = np.einsum('ijkl,l', self.Membership, self.Centers)  # +self.Membership*self.Centers

        ind_non_zero = denominator != 0

        self.weight[ind_non_zero] = denominator[ind_non_zero]
        self.estimate[ind_non_zero] = numerator[ind_non_zero]  # / denominator[ind_non_zero]

        self.weight[(~ind_non_zero)] = self.padding
        self.estimate[(~ind_non_zero)] = self.padding

    def BStep(self, mask=None):
        # bias correction step

        if mask is None:
            mask = self.mask

        mask[~self.mask] = 0
        self.biascorrection.set_info(target=self.image, reference=self.estimate,
                                     weight=self.weight, biasfield=self.biasfield, padding=self.padding,
                                     mask=mask, affine=self.affine, cov_pq=None, use_original=False)

        self.filtered_image = self.image.copy()
        if mask.sum() > 100:
            self.biascorrection.Run()
            self.biascorrection.Apply(self.filtered_image)
            self.filtered_image[~self.mask] = 0
        return


    def _proximity_measure(self, index_, Membership=None, sqr2dist=False):
        # Fuzzy c-means clustering with spatial information for image segmentation

        if Membership is None:
            Membership = self.Membership

        in_out = np.zeros_like(Membership)
        for i in range(self.num_tissues):
            in_out[index_, i] = \
            neighborhood_conv(Membership[..., i][..., None], kerenel_size=3, direction='xyz', sqr2dist=False)[
                index_, 0]

        in_out /= in_out.max()
        return in_out


    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), -1)[..., None]



    def fit(self, progressBar):

        if not hasattr(self, 'Membership'):
            self.Membership = self.atlas_ims.copy()

        degree = 2

        biasf = PolynomialFeatures(degree)  # SplineTransformer(n_knots=2, degree=degree)#
        best_cost = -np.inf
        self.SetBiasField(biasf)
        num_fails = 0
        self.filtered_image = self.image.copy()
        old_cost = np.inf

        i = 0



        while True:
            # if i == 0:
            self.Centers = self.Update_centers()

            old_u = np.copy(self.Membership)
            self.Membership = self.Update_membership()



            cost = np.sum(abs(self.Membership - old_u) > 0.1) / np.prod(self.image[self.mask].shape)

            progressBar.setValue(int(i / (self.max_iter + 1) * 100))
            if self.use_ssim and cost < self.epsilon and self.correct_bias or abs(old_cost - cost) < 1e-6:
                if not self.use_ssim:
                    break
                self.WStep()

                # Apply mapping

                s1 = sobel(self.image)
                s2 = sobel(self.predict())
                fast_method = True
                if fast_method:
                    if self._is2D:
                        cost_ssim, ssim_map = ssim(s1.squeeze() / s1.max(), s2.squeeze() / s2.max(), full=True,
                                           win_size=11, data_range=1)
                        ssim_map = np.expand_dims(ssim_map, np.where(self._imdim)[0][0])
                    else:
                        cost_ssim, ssim_map = ssim(s1 / s1.max(), s2 / s2.max(), full=True,
                                           win_size=11, data_range=1)
                else: # faster
                    ssim_map = ssim3D(s1 / s1.max(), s2 / s2.max(), self.window,
                                  self.window.shape[-1], 1, contrast=False)
                    cost_ssim = ssim_map[self.mask].mean()


                ssim_map = rescale_between_a_b(-ssim_map, -1000, 1000)
                ssim_map[~self.mask] = 0

                if (cost_ssim - best_cost) > 1e-4:
                    print("best SSIM value {}".format(cost_ssim))

                    self.BestCenters = self.Centers.copy()
                    self.BestFilter = self.filtered_image.copy()
                    self.BestMS = self.Membership.copy()


                    best_cost = cost_ssim
                    num_fails = 0
                else:
                    num_fails += 1

                if num_fails > self.max_fail:  # abs(old_cost_ssim - cost_ssim) < 1e-4
                    break
                if num_fails == 0:

                    self.weight = ssim_map  # rescale_between_a_b(sobel(self.image),-1,1) #ssim_map
                    self.BStep(mask=None)
                else:
                    self.filtered_image = self.BestFilter.copy()


            print("Iteration %d : cost = %f" % (i, cost))
            old_cost = cost
            if i > self.max_iter - 1:
                break

            # break
            i += 1

        ### Update with the best parameters
        if self.use_ssim:
            self.Centers = self.BestCenters
            self.filtered_image = self.BestFilter

            self.Membership = self.BestMS


        sortedC = self.Centers.argsort()
        sorted_el = [sortedC[i] for i in range(self.num_tissues)]
        self.Membership = self.Membership[..., sorted_el]
        self.Centers = self.Centers[sorted_el]
        if self.num_tissues == 3:

            if self.type_im=='T1':  # T1
                self.wmlabel = [sortedC[2].item()]
                self.gmlabel = [sortedC[1].item()]
                self.csflabel = [sortedC[0].item()]

        elif self.num_tissues == 2:
            self.wmlabel = [sortedC[1]]
            self.gmlabel = [sortedC[0]]





    def predict(self, use_softmax=False, Membership=None):
        """
        Segment image
        @return:
        """
        if Membership is None:
            Membership = self.Membership
        if use_softmax:
            MM = softmax(Membership, -1)
        else:
            MM = Membership

        sumu = Membership.sum(-1)
        ind_zero = sumu == 0
        maxs = MM.argmax(-1)  # defuzzify
        self.output = maxs + 1
        self.output[ind_zero] = 0
        return self.output




class FCM_pure(FCM):
    """
    Fuzzy c-means clustering with spatial information for image segmentation: 2006
    """
    def __init__(self, image, affine, atlas,
                 image_range, num_tissues, fuzziness,
                 epsilon, max_iter, 
                 padding=0, constraint=False, post_correction=True,mask =None):
        super(FCM_pure).__init__()
        self.biascorrection = BiasCorrection()

        self.mask = mask

        self.estimate = image.copy()  # wstep
        self.weight = image.copy()  # wstep

        self.image = rescale_between_a_b(image, 0, 1000)

        self.image_range = image_range
        self.num_tissues = num_tissues
        self.fuzziness = fuzziness
        self.padding = padding
        self.post_correction = post_correction

        self.epsilon = epsilon
        self.constraint = constraint
        self.max_iter = max_iter

        self.atlas_ims = atlas
        self.shape = image.shape  # image shape
        self.numPixels = image.size
        self.affine = affine

    
     
   





    def Update_memership(self):
        '''Compute weights'''
        # idx = np.arange(self.num_gray)
        # c_mesh, idx_mesh = np.meshgrid(self.Centers, idx)
        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        for i in range(self.num_tissues):
            numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i]) + 1e-7,
                                                        el)


        sumn = numerator.sum(-1)
        numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])

        return numerator


    def Update_centers(self):
        """
        Update center of the clusters
        @return:
        """
        mask = self.mask.copy()

        numerator = np.einsum('il->l', np.expand_dims(self.image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                              self.fuzziness))

        denominator = np.einsum('il->l', pow(self.Membership[mask,:], self.fuzziness))
        ind_non_denom = denominator != 0
        numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]
        numerator[numerator == 0] = 0.00001
        return numerator




    def fit(self, progressBar):
        #if not hasattr(self, 'Membership'):
        #    self.Membership = self.atlas_ims.copy()

        #self.last_seg = self.predict()


        oldd = np.inf
        i = 0
        while True:

            self.Centers = self.Update_centers()

            old_u = np.copy(self.Membership)

            self.Membership = self.Update_memership()

            d = np.sum(abs(self.Membership - old_u) > 0.1) / np.prod(self.image[self.mask].shape)

            print("Iteration %d : cost = %f" % (i, d))
            progressBar.setValue(int(i/(self.max_iter+1)*100))
            if d < self.epsilon  or abs(oldd - d) < 1e-2:
                break

            oldd = d

            i += 1


        self.predict()

    def predict(self, use_softmax=False):
        """
        Segment image
        @return:
        """
        Membership = self.Membership
        if use_softmax:
            MM = softmax(Membership, -1)
        else:
            MM = Membership

        sumu = Membership.sum(-1)
        ind_zero = sumu == 0
        maxs = MM.argmax(-1)  # defuzzify
        self.output = maxs + 1
        self.output[ind_zero] = 0
        return self.output




class Constrained_esFCM(FCM):

    def __init__(self, image, affine, atlas,
                 image_range, num_tissues, fuzziness,
                  epsilon, max_iter,
                 padding=0, constraint=False,
                 tissuelabels=None, correct_bias=True, post_correction=True,
                 mask=None, max_fail=4, use_ssim=True, equalize=False):
        super(Constrained_esFCM, self).__init__()
        self.biascorrection = BiasCorrection()
        self.use_ssim = use_ssim
        self.mask = mask
        self.max_fail = max_fail
        self.window = create_window_3D(11, 1)

        self.estimate = image.copy()  # wstep
        self.weight = image.copy()  # wstep

        if equalize:
            from skimage import exposure
            from melage.utils.utils import histogram_equalization
            self.image = histogram_equalization(image)
        else:
            self.image = image

        self.image_range = image_range
        self.num_tissues = num_tissues
        self.fuzziness = fuzziness
        self.padding = padding
        self.post_correction = post_correction
        self.epsilon = epsilon
        self.constraint = constraint
        self.max_iter = max_iter
        if self.constraint:
            print('Neontal platform is activated')
        else:
            print('Neontal platform is deactivated')
        self.correct_bias = correct_bias

        self.atlas_ims = atlas  # in case atlas image is provided
        self.shape = image.shape  # image shape
        # flatted image shape: (number of pixels,1)

        self.affine = affine
        self.OriginalImage = image
        if self.atlas_ims is not None:
            self._normalizeAtlasCreateMask()
        if self.constraint:
            self.mask_sdf = compute_sdf(mask, bounded=False)
        self.biasfield = 1

        if tissuelabels is not None:
            self.tissuelabels = tissuelabels.astype('int')
        else:
            #
            if self.num_tissues == 3:
                self.tissuelabels = np.array([2, 3, 4])
            else:
                self.tissuelabels = np.zeros(self.num_tissues)
        old = True
        if old:
            self.outlabel = [l[0] for l in np.argwhere(self.tissuelabels == 1)]
            self.csflabel = [l[0] for l in np.argwhere(self.tissuelabels == 2)]
            self.gmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 3)]
            self.wmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 4)]
            self.dgmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 5)]  # non cortical graymatter
            self.venlabel = [l[0] for l in np.argwhere(self.tissuelabels == 6)]

            self.cereblabel = []
            self.bslabel = []  # brain stem
            self.amyglabel = []

        else:

            self.csflabel = [l[0] for l in np.argwhere(self.tissuelabels == 1)]
            self.gmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 2)]
            self.wmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 3)]
            self.outlabel = [l[0] for l in np.argwhere(self.tissuelabels == 4)]
            self.venlabel = [l[0] for l in np.argwhere(self.tissuelabels == 5)]
            self.cereblabel = [l[0] for l in np.argwhere(self.tissuelabels == 6)]
            self.dgmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 7)]  # non cortical graymatter
            self.bslabel = [l[0] for l in np.argwhere(self.tissuelabels == 8)]  # brain stem
            self.amyglabel = [l[0] for l in np.argwhere(self.tissuelabels == 9)]

    def SetBiasField(self, biasfield):
        self.biasfield = biasfield

    def Update_membership(self, constraint=True):
        """
        Updating FCM membership function
        @param final:
        @param membership_p:
        @return:
        """

        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        ###########Compute weight ##########
        a = self.predict()
        uq = np.unique(a)
        uq = [u for u in uq if u != 0]
        pred = np.zeros((*a.shape, self.num_tissues))
        for i, u in enumerate(uq):
            ind = a == u
            pred[ind, i] = 1

        # numstd = np.einsum('ijkl,ijkl->ijk', pred, (self.image[..., None] - self.Centers) ** 2)
        # weight = np.sqrt(numstd)

        if self.atlas_ims is not None:
            weight = self.atlas_ims
            # weight = np.zeros_like(self.atlas_ims)
            # weight = (fuzzy_weights+self.epsilon) / (fuzzy_weights.sum(-1) + self.epsilon)[..., np.newaxis]
            # weight =np.exp(-(self.predict(Membership=self.atlas_ims)[...,np.newaxis] - np.arange(self.num_tissues))**2)
            # weight = (fuzzy_weights + self.epsilon) / (fuzzy_weights.sum(-1) + self.epsilon)[..., np.newaxis]
            # weight=1
        else:
            weight = 1
        for i in range(self.num_tissues):
            if constraint and self.atlas_ims is not None:
                if i == 20 and self.num_tissues == 3:
                    weight = 1
                else:
                    weight = self.atlas_ims[..., i]
                    weight = weight[self.mask]
            else:
                weight = 1
            numerator[self.mask, i] = weight * np.power(abs(self.filtered_image[self.mask] - self.Centers[i]) + 1e-7,
                                                        el)
        # numerator =(((self.filtered_image[..., np.newaxis] - self.Centers)+self.epsilon))**el
        sumn = numerator.sum(-1)
        numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])
        # numerator[~self.mask, :] = 0
        ind_non_zero_maks = self.mask == 1
        # numerator*=rescale_between_a_b(self.weight[..., np.newaxis],0,1)
        # if membership_p is None:
        membership_p = numerator
        if self.atlas_ims is not None:
            mrf_energy = self._proximity_measure(ind_non_zero_maks, self.Membership)
            # mrf_energy*=self.atlas_ims
            # mrf_energy[...,[0,3,4,5]]*=self.atlas_ims[...,[0,3,4,5]]
            # mrf_energy = np.ones_like(numerator)
            # mrf_energy
            # mrf_energy[..., [1, 2]] *= self.atlas_ims[..., [1, 2]]
            # mrf_energy *= self._proximity_measure(ind_non_zero_maks, membership_p)
        else:
            mrf_energy = self._proximity_measure(ind_non_zero_maks, self.Membership)

        numerator *= mrf_energy
        if constraint:
            # sumn = numerator.sum(-1)
            # ind_non_zero = sumn != 0
            # sumn = np.expand_dims(sumn, -1)
            # numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            # pass
            sumn = numerator.sum(-1)
            numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])
            # numerator*=self.atlas_ims
            # numerator = 0.5*(numerator + self.atlas_ims)
            # pass
            # sumn = numerator.sum(-1)
            # ind_non_zero = sumn != 0
            # sumn = np.expand_dims(sumn, -1)
            # numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            # if self.amyglabel is not None:
            # numerator[...,self.amyglabel[0]] = self.atlas_ims[...,self.amyglabel[0]]

            # numerator[..., [3, 5]] *= self.atlas_ims[..., [3, 5]]
            # numerator *= self.atlas_ims
            # numerator[..., [1, 2]] *= self.atlas_ims[..., [1, 2]]
            # mrf_energy[..., [0, 2]] *= self.atlas_ims[..., [0, 2]]
            # numerator[..., [self.csflabel[0]]] = self.atlas_ims[..., [self.csflabel[0]]]
            # self.Membership = np.maximum.reduce([numerator, self.atlas_ims])
            """ 
            sumn = numerator.sum(-1)
            ind_non_zero = sumn != 0
            sumn = np.expand_dims(sumn, -1)
            numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            """
        return numerator

    def Update_membership_old(self):
        '''Compute weights'''
        # idx = np.arange(self.num_gray)
        # c_mesh, idx_mesh = np.meshgrid(self.Centers, idx)
        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        for i in range(self.num_tissues):
            numerator[self.mask, i] = np.power(abs(self.filtered_image[self.mask] - self.Centers[i]) + 1e-7, el)
            # for j in range(self.num_tissues):
            #    numerator[self.mask,i]+=np.power(numi/abs(self.filtered_image[self.mask] - self.Centers[j]), power)
            # denominator[i] = np.sum(abs(self.filtered_image[self.mask] - self.Centers[i]) ** power)
            # if denominator[i]!=0:
            #    numerator[...,i]/= denominator[i]

        # sumn = mrf_energy.sum(-1)
        ##ind_non_zero = sumn != 0
        # sumn = np.expand_dims(sumn, -1)
        # mrf_energy[ind_non_zero, :] /= sumn[ind_non_zero, :]
        ind_non_zero_maks = self.mask == 1

        mrf_energy = self._proximity_measure(ind_non_zero_maks, self.Membership)

        if self.constraint:
            print(self.wmlabel, self.gmlabel)
            if not self.constraint:
                mrf_energy[..., [2]] *= self.atlas_ims[..., [2]]  # just wm
            # mrf_energy[...,[0,2]] *= self.atlas_ims[...,[0,2]]
            numerator *= mrf_energy
            if self.constraint:
                numerator *= self.atlas_ims
                # numerator[..., [0, 1,2,5]] *= self.atlas_ims[..., [0, 1,2,5]]
        else:
            numerator *= mrf_energy
        sumn = numerator.sum(-1)
        ind_non_zero = sumn != 0
        sumn = np.expand_dims(sumn, -1)

        numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
        numerator *= self.atlas_ims

        return numerator

    def Update_centers(self):
        """
        Update center of the clusters
        @return:
        """
        mask = self.mask.copy()

        numerator = np.einsum('il->l',
                              np.expand_dims(self.filtered_image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                     self.fuzziness))
        denominator = np.einsum('il->l', pow(self.Membership[mask], self.fuzziness))
        ind_non_denom = denominator != 0
        numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]

        numerator[numerator == 0] = 0.00001  # for numerical stability
        return numerator

    def WStep(self):
        """
        WStep
        @return:
        """

        input_inp = self.image

        a = self.predict()
        uq = np.unique(a)
        uq = [u for u in uq if u != 0]
        pred = np.zeros((*a.shape, self.num_tissues))
        for i, u in enumerate(uq):
            ind = a == u
            pred[ind, i] = 1

        numstd = np.einsum('ijkl,ijkl->ijk', self.Membership, (input_inp[..., None] - self.Centers) ** 2)
        denominator = np.sqrt(numstd)
        # self.Membership = (self.Membership) / ((self.Membership.sum(-1))[..., np.newaxis])
        # self.Membership[~self.mask, :] = 0
        numerator = np.einsum('ijkl,l', self.Membership, self.Centers)  # +self.Membership*self.Centers

        ind_non_zero = denominator != 0

        self.weight[ind_non_zero] = denominator[ind_non_zero]
        self.estimate[ind_non_zero] = numerator[ind_non_zero]  # / denominator[ind_non_zero]

        self.weight[(~ind_non_zero)] = self.padding
        self.estimate[(~ind_non_zero)] = self.padding

    def BStep(self, mask=None):
        # bias correction step

        if mask is None:
            mask = self.mask

        mask[~self.mask] = 0
        self.biascorrection.set_info(target=self.image, reference=self.estimate,
                                     weight=self.weight, biasfield=self.biasfield, padding=self.padding,
                                     mask=mask, affine=self.affine, cov_pq=None, use_original=False)

        self.filtered_image = self.OriginalImage.copy()
        if mask.sum() > 100:
            self.biascorrection.Run()
            self.biascorrection.Apply(self.filtered_image)
            self.filtered_image[~self.mask] = 0

        return

    def _normalizeAtlasCreateMask(self):
        """
        Normalizing atlas creation
        @return:
        """
        self.atlas_ims[self.atlas_ims < 0] = 0
        mask = self.atlas_ims.sum(-1)
        mask_rep = np.repeat(np.expand_dims(mask, -1), (self.atlas_ims.shape[-1]), -1)
        ind_positive = mask_rep > 0
        self.atlas_ims[ind_positive] /= mask_rep[ind_positive]  # mutual normalize between 0 and 1
        self.atlas_ims[~ind_positive] = 0

    def _csf_gm_bg_correction(self):
        """
        Correction for CSF, Gray Matter and Background
        @return:
        """
        print('Correction for CSF BG GM using nearest neighbour method ...')
        output_image = self.predict()

        gm = output_image.copy()
        label_gm = self.gmlabel[0] + 1
        label_csf = self.csflabel[0] + 1
        ind_csf = gm == label_csf
        ind_gm = gm == label_gm

        gmneighbourscsf = update_according_to_neighbours_conv(output_image, ind_gm, [self.csflabel[0] + 1], sign='+',
                                                              connectivity=6, kernel_size=3)

        gmneighboursOut = update_according_to_neighbours_conv(output_image, ind_gm, [0], sign='+',
                                                              connectivity=6, kernel_size=3)

        gmn = gmneighbourscsf + gmneighboursOut
        ind = (gmneighbourscsf > 0) * (gmneighboursOut > 0)
        self.Membership[(gmn >= 4), label_gm - 1] = 0
        self.Membership[(gmn >= 4), label_csf - 1] = 1
        self.Membership[ind, :] = 0
        self.Membership[ind, label_csf - 1] = 1

    def _proximity_measure(self, index_, Membership=None, sqr2dist=False):
        # Fuzzy c-means clustering with spatial information for image segmentation
        # sFCM
        # invSpacing = 1. / self.spacing
        in_out = np.ones_like(self.Membership)
        if Membership is None:
            Membership = self.Membership

        in_out = np.zeros_like(Membership)
        for i in range(self.num_tissues):
            in_out[index_, i] = \
            neighborhood_conv(Membership[..., i][..., None], kerenel_size=3, direction='xyz', sqr2dist=False)[
                index_, 0]
        # in_out /= 6

        in_out /= in_out.max()
        return in_out

    def _reverse_prob(self, index, label1, label2):
        """
        Reversing the probability
        @param index:
        @param label1:
        @param label2:
        @return:
        """
        tmp1 = self.Membership[index, label1].copy()
        tmp2 = self.Membership[index, label2].copy()
        ind_max = tmp1 > tmp2
        mins_vals = tmp1[~ind_max].copy()
        tmp1[~ind_max] = tmp2[~ind_max]
        tmp2[~ind_max] = mins_vals
        self.Membership[index, label1] = tmp1  # max
        self.Membership[index, label2] = tmp2

    def _connection_between_bg_wm(self):
        seg = self.predict()
        if len(self.outlabel) <= 0:
            return
        index_bg_used = (seg == 0) + (seg == (self.outlabel[0] + 1))
        index_bg_used = index_bg_used > 0
        ind_wm = 0
        for el in self.wmlabel:
            ind_wm += seg == (el + 1)
        ind_wm = ind_wm > 0

        # index of all neighbours plus the center
        ind_wm_neighbours = neighborhood_conv(ind_wm[..., None], kerenel_size=3,
                                              direction='xyz', sqr2dist=False)[..., 0]

        ind_bg_neighbours = \
            neighborhood_conv(index_bg_used[..., None], kerenel_size=3,
                              direction='xyz', sqr2dist=False)[..., 0]
        if len(self.venlabel) > 0:
            ind_non_ventric = seg != (self.venlabel[0] + 1)
            # neigbhour hould less than two should be something else
            ind_common_bg_wm = (ind_wm_neighbours > 0) * (ind_bg_neighbours > 0) * (ind_wm) * (ind_non_ventric)
        else:
            ind_common_bg_wm = (ind_wm_neighbours > 0) * (ind_bg_neighbours > 0) * (ind_wm)
        self.Membership[ind_common_bg_wm, :] = 0
        self.Membership[ind_common_bg_wm, self.gmlabel[0]] = 1

    def _connection_between_csf_wm(self, closing=True):
        """
        Correct the border between CSF and white matter
        @param closing:
        @return:
        """
        seg = self.predict()
        if len(self.outlabel) > 0:
            index_csf_used = (seg == (self.csflabel[0] + 1)).astype('int') + (seg == (self.outlabel[0] + 1)).astype(
                'int')
        else:
            index_csf_used = (seg == (self.csflabel[0] + 1)).astype('int') + (seg == 0).astype('int')
        index_csf_used = index_csf_used > 0
        ind_wm = 0
        for el in self.wmlabel:
            ind_wm += seg == (el + 1)
        ind_wm = ind_wm > 0

        # ind_gm = seg == (self.gmlabel[0] + 1)
        # index_gm_used = ind_gm
        # gms= compute_sdf(ind_gm, bounded=False)

        """

        wms= compute_sdf(ind_wm, bounded=False)
        csfs = compute_sdf(index_csf_used, bounded=False)
        ind_comon_wm_csf =(wms<=2)*(csfs<=0)*(csfs>=-4)*(wms>=0) * (self.mask_sdf > -4)
        self._reverse_probs_new(ind_comon_wm_csf*ind_wm, self.wmlabel[0], self.gmlabel[0])
        self._reverse_probs_new(ind_comon_wm_csf*index_csf_used, self.csflabel[0], self.gmlabel[0])
        return
        """
        # (gms>=0)*(gms<=1)
        # index of all neighbours plus the center
        ind_wm_neighbours = neighborhood_conv(ind_wm[..., None], kerenel_size=3,
                                              direction='xyz', sqr2dist=False)[..., 0]

        ind_csf_neighbours = \
            neighborhood_conv(index_csf_used[..., None], kerenel_size=3,
                              direction='xyz', sqr2dist=False)[..., 0]
        if len(self.venlabel) > 0:
            ind_non_ventric = seg != (self.venlabel[0] + 1)
            # neigbhour hould less than two should be something else
            ind_common_csf_wm = (ind_wm_neighbours > 0) * (ind_csf_neighbours > 0) * (ind_wm) * (
                ind_non_ventric)  # * (self.mask_sdf > -2)
        else:
            ind_common_csf_wm = (ind_wm_neighbours > 0) * (ind_csf_neighbours > 0) * (ind_wm)  # * (self.mask_sdf > -2)
        # ind_common_csf_wm = (ind_wm_neighbours > 0) * (ind_csf_neighbours > 0) * (ind_wm) * (self.mask_sdf > -2)
        ind_common_csf_wm = binary_dilation(ind_common_csf_wm)
        if closing:
            self._reverse_prob(ind_common_csf_wm, self.gmlabel[0], self.wmlabel[0])
        else:
            self._reverse_prob(ind_common_csf_wm, self.csflabel[0], self.wmlabel[0])

    def _remove_extra_wm(self, Membership=None, soft=False):
        """
        Remove extra white matter ...
        @param total:
        @param soft:
        @return:
        """
        if Membership is None:
            Membership = self.Membership

        try:
            seg = self.predict(Membership=Membership)
            index_0 = seg == (self.csflabel[0] + 1)
            if len(self.outlabel) > 0:
                index_0 += seg == (self.outlabel[0] + 1)
            index_0 += seg == 0
            csf_out_zero = 300
            seg[index_0 > 0] = csf_out_zero
            index_csf_out_bg = index_0 > 0
            ind_wm = 0
            for el in self.wmlabel:
                ind_wm += seg == (el + 1)
            ind_wm = ind_wm > 0
            cc, cc_f = LargestCC(ind_wm, 1)
            ind_wm_lcc = 0
            ind_low_wm = np.argwhere(cc_f <= cc_f[cc_f.argsort()[-2]] * 0.05)
            ind_high_wm = np.argwhere(cc_f > cc_f[cc_f.argsort()[-2]] * 0.05)
            if len(ind_high_wm) < len(ind_low_wm):
                ind_non_wm = 0
                for i in ind_high_wm:
                    ind_non_wm += cc == i
                ind_wm_lcc = (~(ind_non_wm > 0))
            else:
                for i in ind_low_wm:
                    ind_wm += cc == i
                ind_wm_lcc = ind_wm > 0
            # ind_wm = seg == (self.wmlabel[0] + 1)
            ind_wm_neighbours = \
                neighborhood_conv(ind_wm_lcc[..., None], kerenel_size=3,
                                  direction='xyz', sqr2dist=False)[..., 0]

            index_csf_new = (ind_wm_neighbours > 1) * (~index_csf_out_bg)

            a1 = axis_based_convolution(ind_wm.astype('int'), kernel_size=3)
            gmlbl = self.gmlabel[0] + 1
            # csflbl = self.csflabel[0] + 1
            csfs_new = np.zeros_like(seg)
            try:
                indx_l = a1[..., 0] > 0
                indx_r = a1[..., 1] > 0
                csfs_new[ind_wm] += ((seg[indx_l] == gmlbl) * (seg[indx_r] == csf_out_zero)) + (
                        (seg[indx_l] == csf_out_zero) * (seg[indx_r] == gmlbl)) + (
                                            (seg[indx_l] == csf_out_zero) * (seg[indx_r] == csf_out_zero))
            except:
                pass
            try:
                indy_l = a1[..., 2] > 0
                indy_r = a1[..., 3] > 0
                csfs_new[ind_wm] += ((seg[indy_l] == gmlbl) * (seg[indy_r] == csf_out_zero)) + (
                        (seg[indy_l] == csf_out_zero) * (seg[indy_r] == gmlbl)) + (
                                            (seg[indy_l] == csf_out_zero) * (seg[indy_r] == csf_out_zero))
            except:
                pass
            try:
                indz_l = a1[..., 4] > 0
                indz_r = a1[..., 5] > 0
                csfs_new[ind_wm] += ((seg[indz_l] == gmlbl) * (seg[indz_r] == csf_out_zero)) + (
                        (seg[indz_l] == csf_out_zero) * (seg[indz_r] == gmlbl)) + (
                                            (seg[indz_l] == csf_out_zero) * (seg[indz_r] == csf_out_zero))
            except:
                pass
            index_csf_new = (csfs_new.astype('int') + index_csf_new.astype('int')) > 0
            for el2 in self.wmlabel:
                Membership[index_csf_new, el2] *= 0.1

            Membership[index_csf_new] = (
                    neighborhood_conv(Membership, kerenel_size=3, direction='xyz', sqr2dist=False)[
                        index_csf_new] / 6)

            return index_csf_new
        except:
            return None

    def _get_index_label(self, seg, label):
        ind_lbl = 0
        for el in label:
            ind_lbl += seg == (el + 1)
        ind_lbl = ind_lbl > 0
        return ind_lbl

    def _binary_fill_holes_lcc(self, index, threshold=3):
        segl, segl_f = LargestCC(index)
        if len(segl_f) > 2:
            argmax_gmf = np.argsort(segl_f)[-2]
        else:
            argmax_gmf = 1
        index_used = (segl != 0) * (segl != argmax_gmf)
        index_used_filled = binary_fill_holes(index_used) > 0
        index_remain = (index_used_filled.astype('int') - index_used.astype('int')) > 0
        segl_f_remove = np.argwhere(segl_f <= threshold)
        segl_f_keep = np.argwhere(segl_f > threshold)

        if len(segl_f_keep) < len(segl_f_remove):
            shouldbe_kept = 0
            for el in segl_f_keep:
                shouldbe_kept += (segl == el).astype('int')
            shouldbe_removed = ~(shouldbe_kept > 0)
        else:
            shouldbe_removed = 0
            for el in segl_f_remove:
                if el == 0:
                    continue
                shouldbe_removed += (segl == el).astype('int')
            if type(shouldbe_removed) == int:
                shouldbe_removed = np.zeros_like(index_used) > 0
            else:
                shouldbe_removed = shouldbe_removed > 0
        return index_remain, shouldbe_removed

    def _neighbours(self, index):
        ind_neighbours = \
            neighborhood_conv(index[..., None], kerenel_size=3,
                              direction='xyz', sqr2dist=False)[..., 0]
        return ind_neighbours

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), -1)[..., None]

    def _remove_common_structures(self, threshold=5):
        seg = self.predict()
        concat1 = []
        concat2 = []
        neigbs = np.zeros((*self.image.shape, self.Membership.shape[-1]))
        if len(self.wmlabel) > 0:
            ind_wm = self._get_index_label(seg, self.wmlabel)
            ind_wm_cc, ind_wm_extra = self._binary_fill_holes_lcc(ind_wm, threshold=threshold)
            neigbs[..., self.wmlabel[0]] = self._neighbours(ind_wm)
            concat1.append(ind_wm_extra)
            concat2.append(ind_wm_cc)
        if len(self.gmlabel) > 0:
            ind_gm = self._get_index_label(seg, self.gmlabel)
            ind_gm_cc, ind_gm_extra = self._binary_fill_holes_lcc(ind_gm, threshold=threshold)
            neigbs[..., self.gmlabel[0]] = self._neighbours(ind_gm)
            concat1.append(ind_gm_extra)
            concat2.append(ind_gm_cc)
        if len(self.csflabel) > 0:
            if len(self.outlabel) > 0:
                ind_csf = self._get_index_label(seg, self.csflabel + self.outlabel)
            else:
                ind_csf = self._get_index_label(seg, self.csflabel)
            ind_csf_cc, ind_csf_extra = self._binary_fill_holes_lcc(ind_csf, threshold=threshold)
            neigbs[..., self.csflabel[0]] = self._neighbours(ind_csf)
            concat1.append(ind_csf_extra)
            concat2.append(ind_csf_cc)
        if len(self.dgmlabel) > 0:
            ind_dgm = self._get_index_label(seg, self.dgmlabel)
            ind_dgm_cc, ind_dgm_extra = self._binary_fill_holes_lcc(ind_dgm, threshold=threshold)
            neigbs[..., self.gmlabel[0]] = self._neighbours(ind_dgm)
            concat1.append(ind_dgm_extra)
            concat2.append(ind_dgm_cc)
        if len(self.venlabel) > 0:
            ind_ven = self._get_index_label(seg, self.venlabel)
            ind_ven_cc, ind_ven_extra = self._binary_fill_holes_lcc(ind_ven, threshold=20)
            neigbs[..., self.venlabel[0]] = self._neighbours(ind_ven)
            concat1.append(ind_ven_extra)
            concat2.append(ind_ven_cc)

        neigbssf = self.softmax(neigbs)
        for indice in concat1:
            self.Membership[indice, :] = neigbssf[indice, :]
        for indice in concat2:
            self.Membership[indice, :] = neigbssf[indice, :]

    def _remove_small_structures(self, label, threshold_percent=0.5, Membership=None):
        if Membership is None:
            Membership = self.Membership
        ### REMOVE SMALL STRUCTURES
        seg = self.predict()
        if len(label) == 0:
            return
        ind_lbl = 0
        for el in label:
            ind_lbl += seg == (el + 1)
        ind_lbl = ind_lbl > 0

        segl, segl_f = LargestCC(ind_lbl, 1)
        if len(segl_f) > 2:
            segl, segl_f = LargestCC(ind_lbl, 1)
            if len(segl_f) > 2:
                argmax_gmf = np.argsort(segl_f)[-2]
            else:
                argmax_gmf = 1
            """

            index_total = 0
            max_v = ind_lbl.sum()/ind_lbl.shape[0]
            ind_remove = np.zeros_like(ind_lbl)
            for sh in range(ind_lbl.shape[0]):
                if ind_lbl[sh,...].sum()<0.05*max_v:
                    continue
                segl, segl_f = LargestCC(ind_lbl[sh,...],1)
                if len(segl_f) > 2:
                    argmax_gmf = np.argsort(segl_f)[-2]
                else:
                    argmax_gmf = 1
                segl_f_remove = np.argwhere(segl_f != segl_f[argmax_gmf])
                shouldbe_removed = 0
                for el in segl_f_remove:
                    if el.all() == 0:
                        continue
                    shouldbe_removed += (segl == el).astype('int')
                ind_remove[sh,...] = shouldbe_removed
            """
            ind_remove = (segl != argmax_gmf) * (segl != 0)
            # segl_f_remove = np.argwhere(ind_remove)
            index_use = ind_remove > 0
            Membership[index_use, label[0]] = 0

            Membership[index_use] = (
                    neighborhood_conv(Membership, kerenel_size=3, direction='xyz', sqr2dist=False)[
                        index_use] / 6)
            return index_use
        else:
            return (ind_lbl * 0) > 0

    def set_tissue_labels(self):
        self.tissuelabels = np.zeros(self.num_tissues)

    def _wm_touch_csf_out(self, outval, csfval, gmval, wmval, ind_wm_gm_csfout, lambda_val):
        # if is a wm voxel that touches (outlier or csf) and gm -> csf, gm

        sum_gm_csf = csfval + gmval
        ind_wm_gm_csfout_no_zero = (sum_gm_csf != 0) * ind_wm_gm_csfout
        # for non zero pixels
        gmval[ind_wm_gm_csfout_no_zero] += (1 - lambda_val) * wmval[ind_wm_gm_csfout_no_zero] * (
                    gmval[ind_wm_gm_csfout_no_zero] / sum_gm_csf[ind_wm_gm_csfout_no_zero])

        csfval[ind_wm_gm_csfout_no_zero] += (1 - lambda_val) * wmval[ind_wm_gm_csfout_no_zero] * (
                    csfval[ind_wm_gm_csfout_no_zero] / sum_gm_csf[ind_wm_gm_csfout_no_zero])

        # for zero pixels
        ind_wm_gm_csfout_zero = (sum_gm_csf == 0) * (ind_wm_gm_csfout)
        gmval[ind_wm_gm_csfout_zero] += (1 - lambda_val) * wmval[ind_wm_gm_csfout_zero] * 0.5
        csfval[ind_wm_gm_csfout_zero] += (1 - lambda_val) * wmval[ind_wm_gm_csfout_zero] * 0.5

        # after that
        wmval[ind_wm_gm_csfout] *= lambda_val
        return [outval, csfval, gmval, wmval]



    def adjust_membership(self, Membership, threshold=None):
        # self.cereblabel = [l[0] for l in np.argwhere(self.tissuelabels == 6)]
        # self.dgmlabel = [l[0] for l in np.argwhere(self.tissuelabels == 7)]  # non cortical graymatter
        # self.bslabel = [l[0] for l in np.argwhere(self.tissuelabels == 8)]  # brain stem
        # self.amyglabel = [l[0] for l in np.argwhere(self.tissuelabels == 9)]
        # self.venlabel = [l[0] for l in np.argwhere(self.tissuelabels == 5)]
        seg = self.predict()
        remain_index = 0
        index_used_filleds = []
        for el in self.dgmlabel + self.amyglabel + self.bslabel + self.venlabel + self.cereblabel:
            index = seg == el + 1
            # index_used_filled = binary_dilation(index)

            segl, segl_f = LargestCC(index, 1)
            if len(segl_f) > 2:
                argmax_gmf = [np.argsort(segl_f)[-2]]
            else:
                argmax_gmf = [1]
            if el == 9:  # Amygdala
                argmax_gmf = [np.argsort(segl_f)[-2], np.argsort(segl_f)[-3]]

            indx = 0
            for maxf in argmax_gmf:
                indx += (segl == maxf)
            index_used = (segl != 0) * indx > 0

            index_used_filled = binary_fill_holes(index_used) > 0
            m = Membership[..., el]
            m[index_used_filled] = np.clip(Membership[index_used_filled, :].max(-1) + 0.1, 0, 1)
            Membership[..., el] = m
            # Membership[index_used_filled, el] = np.clip(Membership[index_used_filled,:].max(-1)+0.1,0,1)

            index_used_filleds.append([index_used_filled, el])

            remain_ind = (index.astype('int') - index_used_filled.astype('int')) > 0
            Membership[remain_ind, el] = 0
            remain_index += remain_ind
        Membership[remain_index > 0] = (
                neighborhood_conv(Membership, kerenel_size=3, direction='xyz', sqr2dist=False)[
                    remain_index > 0] / 6)

        return remain_index > 0, index_used_filleds

    def _reverse_probs_new(self, ind_revise, label1, label2, exclude_largest=False):
        if exclude_largest:
            segl, segl_f = LargestCC(ind_revise, 1)
            if len(segl_f) > 2:
                argmax_gmf = [np.argsort(segl_f)[-2]]
            else:
                argmax_gmf = [1]
            indx = 0
            for maxf in argmax_gmf:
                indx += (segl == maxf)
            index_used = (segl != 0) * indx > 0
            remain_ind = (ind_revise.astype('int') - index_used.astype('int')) > 0
        else:
            remain_ind = ind_revise
        indices = np.where(remain_ind > 0)

        m = self.Membership[indices[0], indices[1], indices[2], label1].copy()
        m2 = self.Membership[indices[0], indices[1], indices[2], label2]

        self.Membership[indices[0], indices[1], indices[2], label2] = m
        self.Membership[indices[0], indices[1], indices[2], label1] = m2

    def correct_wm_csf_gm(self):
        summation = 0
        seg = self.predict()
        ind_wm = seg == self.wmlabel[0] + 1
        # ind_csf = seg == self.csflabel[0] + 1
        ind_gm = seg == self.gmlabel[0] + 1
        gms = compute_sdf(ind_gm, bounded=False)
        spacing = 0.5  # self.header['pixdim'][1]
        segl, segl_f = LargestCC(ind_wm, 1)
        if len(segl_f) > 2:
            argmax_gmf = [np.argsort(segl_f)[-2], np.argsort(segl_f)[-3]]
        else:
            argmax_gmf = 1
        ind_wm2 = (segl != argmax_gmf[0]) * (segl > 0) * (segl != argmax_gmf[1])
        wm2 = compute_sdf(ind_wm2, bounded=False)
        # contamination of white matter in gray matter
        ind_revise_wm_gm = (((gms >= 0) * (gms <= 1) * (wm2 <= 1) * ind_wm))
        summation += ind_revise_wm_gm.sum()
        if summation != 0:
            self._reverse_probs_new(ind_revise_wm_gm, self.wmlabel[0], self.csflabel[0])
        seg = self.predict()
        ind_wm = seg == self.wmlabel[0] + 1

        cc, cc_f = LargestCC(ind_wm, 1)
        if len(cc_f) > 2:
            sorted_f = np.argsort(cc_f)
            argmax_g = [sorted_f[-2], sorted_f[-3]]
        else:
            argmax_g = [0, 0]
        # ind_wm_larg = ((cc==argmax_g[0]).astype('int')+(cc==argmax_g[1]).astype('int'))>0
        # remain_wm = (ind_wm.astype('int')-ind_wm_larg.astype('int'))>0
        if len(self.csflabel) > 0 and 1 > 2:
            ind_ = (seg == self.csflabel[0] + 1).astype('int')
            if len(self.outlabel) > 0:
                ind_ += (seg == self.outlabel[0] + 1).astype('int')
            ind_csf = (ind_) > 0
            # ind_gm = seg == self.gmlabel[0] + 1

            ind_rest = (seg != 0).astype('int') - (ind_csf.astype('int') + ind_gm.astype('int') + ind_wm.astype('int'))
            # gms = compute_sdf(ind_gm, bounded=False)
            wms = compute_sdf(ind_wm, bounded=False)
            csfs = compute_sdf(ind_csf, bounded=False)
            rest = compute_sdf(ind_rest, bounded=False)
            ind_revise_wm_csf = ((csfs <= 0) * (
                    (csfs >= -3) * (wms >= 0) *
                    (wms <= 2) * (self.mask_sdf > -10) * (self.mask_sdf <= -3) * (rest > 5) * ind_csf))
            ind_revise_wm_csf = (ind_revise_wm_csf.astype('int') - ind_revise_wm_gm.astype('int')) > 0
            ind_revise_wm_csf *= (seg == self.csflabel[0] + 1)
            self._reverse_probs_new(ind_revise_wm_csf * ind_wm, self.wmlabel[0], self.csflabel[0])
            self._reverse_probs_new(ind_revise_wm_csf * ind_csf, self.csflabel[0], self.gmlabel[0])
            summation += ind_revise_wm_csf.sum()
            # self._reverse_probs_new(ind_revise_wm_csf, self.wmlabel[0], self.csflabel[0])
        return summation

    def _correction_wm(self):
        cc, cc_f = LargestCC(self.predict() == self.wmlabel[0] + 1, 1)
        if len(cc_f) > 2:
            argm = cc_f.argsort()[-2]
            argms = np.argwhere(cc_f > 0.05 * cc_f[argm]).squeeze()
            for a in argms:
                cc[cc == a] = 0
            A = self.Membership[..., self.wmlabel]
            A[cc > 0] = 0
            self.Membership[..., self.wmlabel] = A

    def fit(self, progressBar):

        #    ind_common_csf_dgm = ((self.atlas_ims[..., 3] > 0.1) * (self.atlas_ims[..., 0] > 0.1))
        #    self.atlas_ims[ind_common_csf_dgm, self.csflabel[0]] *= 0.1
        if not hasattr(self, 'Membership') or self.constraint:
            self.Membership = self.atlas_ims.copy()

        degree = 2
        max_degree = 2
        biasf = PolynomialFeatures(degree)  # SplineTransformer(n_knots=2, degree=degree)#
        best_cost = -np.inf
        self.SetBiasField(biasf)
        num_fails = 0
        self.filtered_image = self.image.copy()
        old_cost = np.inf
        old_cost_ssim = np.inf
        i = 0
        # self.Centers = self.Update_centers()

        # self.Membership = self.Update_membership(constraint=False)
        # self.WStep()
        # self.filtered_image = rescale_between_a_b(self.weight, 0, 1000).copy()
        # self.filtered_image = rescale_between_a_b(self.filtered_image+gaussian_filter(self.filtered_image,2),0,1000)
        # self.filtered_image[~self.mask]=0
        # a = neighborhood_conv(self.filtered_image[..., np.newaxis]).squeeze()
        ssim_map = None
        # alphas = np.linspace(0.3,1,self.max_iter)
        while True:
            # if i == 0:
            self.Centers = self.Update_centers()

            old_u = np.copy(self.Membership)
            self.Membership = self.Update_membership()

            if self.constraint:
                self.adjust_membership(self.Membership)

            cost = np.sum(abs(self.Membership - old_u) > 0.1) / np.prod(self.image[self.mask].shape)

            progressBar.setValue(int(i / (self.max_iter + 1) * 100))
            if self.use_ssim and cost < self.epsilon and self.correct_bias or abs(old_cost - cost) < 1e-6:
                if not self.use_ssim:
                    break
                self.WStep()

                # Apply mapping

                s1 = sobel(self.image)
                s2 = sobel(self.predict())
                #ssim_map = ssim3D(s1 / s1.max(), s2 / s2.max(), self.window,
                #                  self.window.shape[-1], 1, contrast=True)
                #cost_ssim = ssim_map[self.mask].mean()
                from skimage.metrics import structural_similarity as ssim
                cost_ssim, ssim_map = ssim(s1 / s1.max(), s2 / s2.max(), full=True,
                                           win_size=11)

                ssim_map = rescale_between_a_b(-ssim_map, -1000, 1000)
                ssim_map[~self.mask] = 0

                if (cost_ssim - best_cost) > 1e-4:
                    print("best SSIM value {}".format(cost_ssim))
                    summation_new = np.inf

                    if self.constraint:
                        summation_new = self.correct_wm_csf_gm()
                        # changed_index = self._remove_extra_wm(Membership=self.Membership)
                        # if changed_index is not None:
                        #    self.atlas_ims[changed_index, :] = self.Membership[changed_index, :]
                        changed_index, index_changed_el = self.adjust_membership(self.Membership, threshold=None)
                        for [indx, el] in index_changed_el:
                            self.atlas_ims[indx, el] = self.Membership[indx, el]
                        self.atlas_ims[changed_index, :] = self.Membership[changed_index, :]
                        # changed_index = self._remove_small_structures(self.gmlabel, Membership=self.Membership)
                        # self.atlas_ims[changed_index, :] = self.Membership[changed_index, :]
                        # self.atlas_ims[...,0] = self.Membership[...,0].copy()
                    self.BestCenters = self.Centers.copy()
                    self.BestFilter = self.filtered_image.copy()
                    self.BestMS = self.Membership.copy()
                    if self.atlas_ims is not None:
                        self.Membership = self.atlas_ims.copy()

                    best_cost = cost_ssim
                    num_fails = 0
                else:
                    num_fails += 1

                if num_fails > self.max_fail:  # abs(old_cost_ssim - cost_ssim) < 1e-4
                    break
                if num_fails == 0:

                    self.weight = ssim_map  # rescale_between_a_b(sobel(self.image),-1,1) #ssim_map
                    self.BStep(mask=None)
                else:
                    self.filtered_image = self.BestFilter.copy()
                    # self.Membership = self.BestMS.copy()

                old_cost_ssim = cost_ssim

            #    self.huiPVCorrection()
            # self._remove_common_structures(threshold=3)

            print("Iteration %d : cost = %f" % (i, cost))
            old_cost = cost
            if i > self.max_iter - 1:
                break

            # break
            i += 1

        ### Update with the best parameters
        if self.use_ssim:
            self.Centers = self.BestCenters
            self.filtered_image = self.BestFilter

            self.Membership = self.BestMS

        # if degree==max_degree and not self.constraint:
        if not self.constraint:
            sortedC = self.Centers.argsort()
            sorted_el = [sortedC[i] for i in range(self.num_tissues)]
            self.Membership = self.Membership[..., sorted_el]
            self.Centers = self.Centers[sorted_el]
            if self.num_tissues == 3:

                if len(self.csflabel) > 0:  # T1
                    self.wmlabel = [sortedC[2].item()]
                    self.gmlabel = [sortedC[1].item()]
                    self.csflabel = [sortedC[0].item()]

            elif self.num_tissues == 2:
                self.wmlabel = [sortedC[1]]
                self.gmlabel = [sortedC[0]]
        else:
            pass
            # if self.num_tissues==3:
            # sortedC = self.Centers.argsort()
            # self.Membership = self.Membership[..., [sortedC[0], sortedC[2], sortedC[1]]]
            # self.Centers = self.Centers[[sortedC[0], sortedC[2], sortedC[1]]]
            # if len(self.csflabel) > 0:  # T1
            # self.wmlabel = [sortedC[2].item()]
            # self.gmlabel = [sortedC[1].item()]
            # self.csflabel = [sortedC[0].item()]

        #self.post_correction = True
        print('post correction is {}'.format(self.post_correction))
        if self.post_correction:




            ### REMOVE SMALL STRUCTURES FROM WM
            self.adjust_membership(self.Membership)

            self._remove_small_structures(self.dgmlabel)

            self.Membership = adjust_common_structures(self.Membership, threshold=None)
            summation_old = 0
            summation_new = np.inf

            if self.constraint and self.num_tissues > 4:
                self._correction_wm()
                self._connection_between_csf_wm(closing=True)
            else:
                self._correction_wm()
                # self._connection_between_bg_wm()

        self.predict()

    def predict(self, use_softmax=False, Membership=None):
        """
        Segment image
        @return:
        """
        if Membership is None:
            Membership = self.Membership
        if use_softmax:
            MM = softmax(Membership, -1)
        else:
            MM = Membership

        sumu = Membership.sum(-1)
        ind_zero = sumu == 0
        maxs = MM.argmax(-1)  # defuzzify
        self.output = maxs + 1
        self.output[ind_zero] = 0
        return self.output

