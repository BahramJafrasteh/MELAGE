import numpy as np
from scipy.special import logsumexp
from scipy import linalg
from sklearn.cluster import KMeans
"""
GMM has been used from
https://atavory.github.io/ibex/_modules/sklearn/mixture/gaussian_mixture.html
with some modification
"""

def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)




###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)



class GaussianMixture(object):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
    #@_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
     #   super().__init__(
      #      n_components=n_components, tol=tol, reg_covar=reg_covar,
       #     max_iter=max_iter, n_init=n_init, init_params=init_params,
        #    random_state=random_state, warm_start=warm_start,
         #   verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _estimate_gaussian_covariances_full(self, resp, X, nk, means, reg_covar):
        """Estimate the full covariance matrices.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::n_features + 1] += reg_covar
        return covariances



    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features)
            The covariance vector of the current components.
        """
        #X = X * self._counts
        avg_X2 = np.dot((resp).T, (X * X*self._counts) ) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X*self._counts) / nk[:, np.newaxis]
        return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data array.

        resp : array-like of shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        nk : array-like of shape (n_components,)
            The numbers of data samples in the current components.

        means : array-like of shape (n_components, n_features)
            The centers of the current components.

        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = (resp*self._counts).sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X*self._counts)/nk[:,np.newaxis]
        covariances = {"diag": self._estimate_gaussian_covariances_diag
                       }[covariance_type](resp, X, nk, means, reg_covar)
        return nk, means, covariances

    def _compute_precision_cholesky(self, covariances, covariance_type):
        """Compute the Cholesky decomposition of the precisions.

        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")

        if covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                try:
                    cov_chol = linalg.cholesky(covariance, lower=True)
                except linalg.LinAlgError:
                    raise ValueError(estimate_precision_error_message)
                precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                             np.eye(n_features),
                                                             lower=True).T
        elif covariance_type == 'tied':
            _, n_features = covariances.shape
            try:
                cov_chol = linalg.cholesky(covariances, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                      lower=True).T
        else:
            if np.any(np.less_equal(covariances, 0.0)):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1. / np.sqrt(covariances)
        return precisions_chol

    ###############################################################################
    # Gaussian mixture probability estimators
    def _compute_log_det_cholesky(self, matrix_chol, covariance_type, n_features):
        """Compute the log-det of the cholesky decomposition of matrices.
        """

        if covariance_type == 'diag':
            log_det_chol = (np.sum(np.log(matrix_chol), axis=1))


        return log_det_chol

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol, covariance_type):
        """Estimate the log Gaussian probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        means : array-like of shape (n_components, n_features)

        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)

        covariance_type : {'full', 'tied', 'diag', 'spherical'}

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        _, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = self._compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features)


        if covariance_type == 'diag':
            precisions = precisions_chol ** 2
            #if self._counts is not None:
             #   X = X*self._counts
            log_prob = (np.sum((means ** 2 * precisions), 1) -
                        2. * np.dot(X, (means * precisions).T) +
                        np.dot(X ** 2, precisions.T))
            if self._counts is not None:
                log_prob *= self._counts

        return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def fit(self, X, y=None, counts = None, total_data_kmeans=None):
        """Estimate model parameters with the EM algorithm.
        """
        if counts is not None:
            self._counts = counts.reshape(-1,1)
            self._total_data_kmeans = total_data_kmeans
        self.fit_predict(X, y, counts)
        self._counts = None
        return self

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        """


        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time



    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t ll change %.5f" % (
                    n_iter, cur_time - self._iter_prev_time, diff_ll))
                self._iter_prev_time = cur_time


    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            km = KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(self._total_data_kmeans)
            label = km.predict(X)
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" %
                  (self.converged_, time() - self._init_prev_time, ll))


    def fit_predict(self, X, y=None, counts = None):
        """Estimate model parameters using X and predict the labels for X.
        """
        #X = _check_X(X, self.n_components, ensure_min_samples=2)
        #self._check_n_features(X, reset=True)
        #self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = np.random.RandomState(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = (-np.infty if do_init else self.lower_bound_)

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X, counts=counts)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            print('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1))

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X, counts=None):
        """E step.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm.sum()/self._counts.sum(), log_resp

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        #if self._counts is not None:
         #   weighted_log_prob = self._counts * weighted_log_prob
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples= self._counts.sum()#X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = self._compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples = self._counts.sum()#X.shape
        self.weights_, self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = self._compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
