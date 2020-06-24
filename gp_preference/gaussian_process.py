import numpy as np
from numpy.random import RandomState
from scipy.stats import norm
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_data


class GPPairwise:
    """
    Gaussian process with a probit likelihood for pairwise comparisons.
    """
    def __init__(self, num_objectives, std_noise=0.01, kernel_width=0.15, prior_mean_type='zero', seed=None):
        """
        :param num_objectives:      number of objectives of input for utility function we want to approximate
        :param std_noise:           the standard deviation of the normal distributed noise we assume for the utility function
        :param prior_mean_type:     prior mean function type (zero/linear), default is zero
        :param kernel_width:        parameter for kernel width, deafult is 0.2
        :param seed:                seed for random state
        """
        self.num_objectives = num_objectives
        self.std_noise = std_noise
        self.kernel_width = kernel_width
        self.prior_mean_type = prior_mean_type
        self.random_state = RandomState(seed)

        # variables for the observed data
        self.datapoints = None
        self.comparisons = None

        # approximate utility values of the datapoints
        self.utility_vals = None

        # covariance matrix of datapoints
        self.cov_mat = None
        self.cov_mat_inv = None

        # hessian (second deriv) of the pairwise likelihoods for observed data
        self.hess_likelihood = None
        self.hess_likelihood_inv = None

        # needed for predictive distribution (cov - hess_likelihood_inv)^(-1)
        self.pred_cov_factor = None

    def sample(self, sample_points):
        """
        Get a sample from the current GP at the given points.
        :param sample_points:   the points at which we want to take the sample
        :return:                the values of the GP sample at the input points
        """
        # bring sample points in right shape
        sample_points = utils_data.format_data(sample_points, self.num_objectives)

        # get the mean and the variance of the predictive (multivariate gaussian) distribution at the sample points
        mean, var = self.get_predictive_params(sample_points, pointwise=False)

        # sample from the multivariate gaussian with the given parameters
        f_sample = self.random_state.multivariate_normal(mean, var, 1)[0]

        return f_sample

    def get_predictive_params(self, x_new, pointwise):
        """
        Returns the predictive parameters (mean, variance) of the Gaussian distribution
        at the given datapoints
        :param x_new:    the points for which we want the predictive params
        :param pointwise:       whether we want pointwise variance or the entire covariance matrix
        :return:
        """
        # bring input points into right shape
        x_new = utils_data.format_data(x_new, self.num_objectives)

        # if we don't have any data yet, use prior GP to make predictions
        if self.datapoints is None or self.utility_vals is None:
            pred_mean, pred_var = self._evaluate_prior(x_new)

        # otherwise compute predictive mean and covariance
        else:
            cov_xnew_x = self._cov_mat(x_new, self.datapoints, noise=False)
            cov_x_xnew = self._cov_mat(self.datapoints, x_new, noise=False)
            cov_xnew = self._cov_mat(x_new, noise=False)
            pred_mean = self.prior_mean(x_new) + np.dot(np.dot(cov_xnew_x, self.cov_mat_inv), (self.utility_vals - self.prior_mean(self.datapoints)))
            pred_var = cov_xnew - np.dot(np.dot(cov_xnew_x, self.pred_cov_factor), cov_x_xnew)

        if pointwise:
            pred_var = pred_var.diagonal()

        return pred_mean, pred_var

    def update(self, dataset):
        """
        Update the Gaussian process using the given data
        :param dataset:
        :return:        """
        self.datapoints = dataset.datapoints
        self.comparisons = dataset.comparisons

        # compute the covariance matrix given the new datapoints
        self.cov_mat = self._cov_mat(self.datapoints)
        self.cov_mat_inv = np.linalg.inv(self.cov_mat)

        # compute the map estimate of f
        self.utility_vals = self._compute_posterior()

        # compute the hessian of the likelihood given f_MAP
        self.hess_likelihood = self._compute_hess_likelihood()
        try:
            self.hess_likelihood_inv = np.linalg.inv(self.hess_likelihood)
        except:
            self.hess_likelihood_inv = np.linalg.pinv(self.hess_likelihood)

        self.pred_cov_factor = np.linalg.inv(self.cov_mat - self.hess_likelihood_inv)

    def _evaluate_prior(self, input_points):
        """
        Given some datapoints, evaluate the prior
        :param input_points:    input datapoints at which to evaluate prior
        :return:                predictive mean and covariance at the given inputs
        """
        pred_mean = self.prior_mean(input_points)
        num_inputs = input_points.shape[0]
        pred_cov = self._kernel(np.repeat(input_points, num_inputs, axis=0),
                                np.tile(input_points, (num_inputs, 1))).reshape((num_inputs, num_inputs))
        return pred_mean, pred_cov

    def _cov_mat(self, x1, x2=None, noise=True):
        """
        Covariance matrix for preference data using the kernel function.
        :param x1:      datapoints for which to compute covariance matrix
        :param x2:      if None, covariance matrix will be square for the input x1
                        if not None, covariance will be between x1 (rows) and x2 (cols_
        :param noise:   whether to add noise to the diagonal of the covariance matrix
        :return:
        """
        if x2 is None:
            x2 = x1
        else:  # if x1 != x2 we don't add noise!
            noise = False

        x1 = utils_data.format_data(x1, self.num_objectives)
        x2 = utils_data.format_data(x2, self.num_objectives)

        cov_mat = self._kernel(np.repeat(x1, x2.shape[0], axis=0), np.tile(x2, (x1.shape[0], 1)))
        cov_mat = cov_mat.reshape((x1.shape[0], x2.shape[0]))

        if noise:
            cov_mat += self.std_noise ** 2 * np.eye(cov_mat.shape[0])

        return cov_mat

    def prior_mean(self, x):
        """
        Prior mean function
        :param x:   num_datapoints * num_objectives
        :return:
        """
        x = utils_data.format_data(x, self.num_objectives)
        m = np.zeros(x.shape[0])
        if self.prior_mean_type == 'linear':
            m += np.sum(x, axis=1) / self.num_objectives
        else:
            TypeError('Prior mean type not understood.')
        return m

    def _kernel(self, x1, x2):
        x1 = utils_data.format_data(x1, self.num_objectives)
        x2 = utils_data.format_data(x2, self.num_objectives)
        k = 0.8**2 * np.exp(-(1. / (2. * (self.kernel_width ** 2))) * np.linalg.norm(x1 - x2, axis=1) ** 2)
        return k

    def _compute_hess_likelihood(self, z=None):
        """
        Compute the hessian of the likelihood given utility values f
        :return:
        """
        if z is None:
            # compute z
            f_winner = np.array([self.utility_vals[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
            f_loser = np.array([self.utility_vals[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
            z = (f_winner - f_loser) / (np.sqrt(2) * self.std_noise)

        z_logpdf = norm.logpdf(z)
        z_logcdf = norm.logcdf(z)

        # initialise with zeros
        lambda_mat = np.zeros((self.datapoints.shape[0], self.datapoints.shape[0]))

        # build up diagonal for pairs (xi, xi)
        diag_arr = np.array([self._compute_hess_likelihood_entry(m, m, z, z_logpdf, z_logcdf) for m in range(self.datapoints.shape[0])])
        np.fill_diagonal(lambda_mat, diag_arr)  # happens in-place

        # go through the list of comparisons collected so far and update lambda
        for k in range(self.comparisons.shape[0]):
            m = self.comparisons[k, 0]  # winner
            n = self.comparisons[k, 1]  # loser
            lambda_mat[m, n] = self._compute_hess_likelihood_entry(m, n, z, z_logpdf, z_logcdf)
            lambda_mat[n, m] = self._compute_hess_likelihood_entry(n, m, z, z_logpdf, z_logcdf)

        # add jitter term to make lambda positive definite for computational stability
        lambda_mat += np.eye(self.datapoints.shape[0]) * 0.01

        return lambda_mat

    def _compute_hess_likelihood_entry(self, m, n, z, z_logpdf, z_logcdf):
        """
        Get a single entry for the Hessian matrix at indices (m,n)
        :param m:
        :param n:
        :param f:
        :return:
        """
        h_x_m = np.array(self.comparisons[:, 0] == m, dtype=int) - np.array(self.comparisons[:, 1] == m, dtype=int)
        h_x_n = np.array(self.comparisons[:, 0] == n, dtype=int) - np.array(self.comparisons[:, 1] == n, dtype=int)
        p = h_x_m * h_x_n * (np.exp(2.*z_logpdf - 2.*z_logcdf) + z * np.exp(z_logpdf - z_logcdf))
        c = - np.sum(p) / (2 * self.std_noise**2)
        return c

    def _compute_posterior(self):
        """ Approximate the posterior distribution """

        converged = False
        try_no = 0

        f_map = None

        # using Newton-Raphson, approximate f_MAP
        while not converged and try_no < 1:

            # randomly initialise f_map
            f_map = self.random_state.uniform(0., 1., self.datapoints.shape[0])
            # f_map = np.zeros(gp.datapoints.shape[0])

            for m in range(100):

                # compute z
                f_winner = np.array([f_map[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
                f_loser = np.array([f_map[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
                z = (f_winner - f_loser) / (np.sqrt(2) * self.std_noise)
                z_logpdf = norm.logpdf(z)
                z_logcdf = norm.logcdf(z)

                # compute b
                h_j = np.array([np.array(self.comparisons[:, 0] == j, dtype=int) - np.array(self.comparisons[:, 1] == j,
                                                                                          dtype=int) for j in
                                range(self.datapoints.shape[0])])
                b = np.sum(h_j * np.exp(z_logpdf - z_logcdf), axis=1) / (np.sqrt(2) * self.std_noise)

                # compute gradient g
                g = - np.dot(self.cov_mat_inv, (f_map - self.prior_mean(self.datapoints))) + b

                # compute approximation of the hessian of the posterior
                hess_likelihood = self._compute_hess_likelihood(z)
                hess_posterior = - self.cov_mat_inv + hess_likelihood
                try:
                    hess_posterior_inv = np.linalg.inv(hess_posterior)
                except:
                    hess_posterior_inv = np.linalg.pinv(hess_posterior)

                # perform update
                update = np.dot(hess_posterior_inv, g)
                f_map -= update

                # stop criterion
                if np.linalg.norm(update) < 0.0001:
                    converged = True
                    break

            if not converged:
                print("Did not converge.")
                try_no += 1

        return f_map


class GPPairwiseDerivative(GPPairwise):
    """
    Gaussian process for pairwise comparisons plus virtual derivatives
    """
    def __init__(self, *args, **kwargs):
        GPPairwise.__init__(self, *args, **kwargs)

        self.datapoints_deriv = None                    # derivative points
        self.x_dx = np.empty((0, self.num_objectives))  # datapoints + derivative points
        self.nu = 10e+6                                 # hyperparameter for probit likelihood

    def prior_mean_deriv(self, x):
        """
        Derivative of prior mean function
        :param x:
        :return:
        """
        x = utils_data.format_data(x, self.num_objectives)
        if self.prior_mean_type == 'zero':
            dm = np.zeros(x.shape[0])
        elif self.prior_mean_type == 'linear':
            dm = np.ones(x.shape[0]) / self.num_objectives
        else:
            TypeError('Prior mean type not understood.')
        return dm

    def update(self, dataset, dx):
        """ pairwise dataset and virtual derivative points """
        self.datapoints = dataset.datapoints
        self.comparisons = dataset.comparisons

        # x-values at which we make virtual derivative observations
        self.datapoints_deriv = utils_data.format_data(dx, self.num_objectives)

        # x-values for real observations and derivative observations
        self.x_dx = np.concatenate((self.datapoints, self.datapoints_deriv))

        # update the covariance matrix
        self.cov_mat = self._cov_mat_incl_deriv(self.datapoints, self.datapoints_deriv)
        self.cov_mat_inv = np.linalg.inv(self.cov_mat)

        # update the map estimate of f
        self.utility_vals = self._compute_posterior()

    def get_predictive_params(self, x_new, pointwise):

        # make sure data is in right shape
        x_new = utils_data.format_data(x_new, self.num_objectives)

        # if we don't have any data yet, use prior GP to make predictions
        if self.x_dx.shape[0] == 0 or self.utility_vals is None:
            pred_mean, pred_var = self._evaluate_prior(x_new)

        # otherwise compute predictive mean and covariance
        else:
            cov_in_data = self._cov_mat_incl_deriv(x1=x_new, x2=self.datapoints, dx2=self.datapoints_deriv)
            cov_data_in = self._cov_mat_incl_deriv(x1=self.datapoints, dx1=self.datapoints_deriv, x2=x_new)
            cov_lamb_inv = np.linalg.inv(self.cov_mat - self.lambda_mat_inv)
            pred_mean = self.prior_mean(x_new) + np.dot(np.dot(cov_in_data, self.cov_mat_inv), (self.utility_vals - self.prior_joint_mean(self.datapoints, self.datapoints_deriv)))
            pred_var = self._kernel(x_new, x_new) - np.dot(np.dot(cov_in_data, cov_lamb_inv), cov_data_in)
        if pointwise:
            pred_var = pred_var.diagonal()

        return pred_mean, pred_var

    def _cov_mat_incl_deriv(self, x1, dx1=None, x2=None, dx2=None, noise=True):

        # make sure that dx1 XOR dx2 is true
        assert (dx1 is None and dx2 is not None) or (dx1 is not None and dx2 is None)

        # upper left
        cov_mat_x = self._cov_mat(x1, x2, noise)

        # upper right and lower left
        if (dx1 is not None) and (x2 is None) and (dx2 is None):
            cov_mat_x_dx = self._cov_mat_x_dx(x1, dx1)
            cov_mat_dx_x = self._cov_mat_dx_x(dx1, x1)
            cov_mat_dx = self._cov_mat_dx(dx1, noise)
            cov_mat = np.vstack((np.hstack((cov_mat_x, cov_mat_x_dx)), np.hstack((cov_mat_dx_x, cov_mat_dx))))
        elif (dx1 is None) and (x2 is not None) and (dx2 is not None):
            cov_mat_x_dx = self._cov_mat_x_dx(x1, dx2)
            cov_mat = np.hstack((cov_mat_x, cov_mat_x_dx))
        elif (dx1 is not None) and (x2 is not None) and (dx2 is None):
            cov_mat_dx_x = self._cov_mat_dx_x(dx1, x2)
            cov_mat = np.vstack((cov_mat_x, cov_mat_dx_x))
        else:
            raise RuntimeError()

        return cov_mat

    def _cov_mat_x_dx(self, x, dx):
        if len(x) == 0 or len(dx) == 0:
            cov_mat_x_dx = np.empty((len(x), len(dx)*self.num_objectives))
        else:
            cov_mat_x_dx = self._kernel_x_dx(np.repeat(x, dx.shape[0], axis=0), np.tile(dx, (x.shape[0], 1)))
            cov_mat_x_dx = cov_mat_x_dx.reshape((x.shape[0], -1))
        return cov_mat_x_dx

    def _cov_mat_dx_x(self, dx, x):
        if len(dx) == 0 or len(x) == 0:
            cov_mat_dx_x = np.empty((len(dx)*self.num_objectives, len(x)))
        else:
            cov_mat_dx_x = self._kernel_dx_x(np.repeat(dx, x.shape[0], axis=0), np.tile(x, (dx.shape[0], 1)))
            cov_mat_dx_x = cov_mat_dx_x.reshape((-1, x.shape[0]))
        return cov_mat_dx_x

    def _cov_mat_dx(self, dx, noise=True):

        kernel_dx_dx = self._kernel(np.repeat(dx, dx.shape[0], axis=0), np.tile(dx, (dx.shape[0], 1))).reshape((dx.shape[0], dx.shape[0]))
        cov_mat_dx = 1./(self.kernel_width**2) * np.tile(kernel_dx_dx, (self.num_objectives, self.num_objectives))
        indic = np.tile(np.eye(self.num_objectives), (dx.shape[0], dx.shape[0]))
        # wdim1 = np.repeat(np.tile(dx.T, (dx.shape[0], 1)), self.num_objectives, axis=1)
        wdim1 = np.tile(dx.flatten(), (dx.shape[0]*self.num_objectives, 1))
        # wdim2 = np.tile(dx.T, (self.num_objectives, self.num_objectives))
        wdim2 = np.repeat(np.tile(dx.T, (dx.shape[0], 1)), self.num_objectives, axis=1)
        cov_mat_dx *= indic - 1./(self.kernel_width**2) * (wdim1 - wdim2) * (wdim1.T - wdim2.T)

        if noise:
            cov_mat_dx += self.std_noise ** 2 * np.eye(cov_mat_dx.shape[0])

        return cov_mat_dx

    def _kernel_x_dx(self, x, dx):
        return 1./(self.kernel_width**2) * (x.flatten() - dx.flatten()) * np.tile(self._kernel(x, dx), self.num_objectives)

    def _kernel_dx_x(self, x1, x2):
        return - self._kernel_x_dx(x1, x2)

    def prior_joint_mean(self, x, dx):
        return np.hstack((self.prior_mean(x), np.repeat(self.prior_mean_deriv(dx), self.num_objectives)))

    def _compute_hess_likelihood_x_dx(self, f_df, nu=10e-8):

        f = f_df[:self.datapoints.shape[0]]
        df = f_df[self.datapoints.shape[0]:]

        # Hessian of likelihoods
        hess_likelihood_x_dx = np.zeros((f_df.shape[0], f_df.shape[0]))

        # Hessian of pairwise likelihood
        if f.shape[0] > 0:
            # compute z
            f_winner = np.array([f[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
            f_loser = np.array([f[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
            z = (f_winner - f_loser) / (np.sqrt(2) * self.std_noise)
            z_logpdf = norm.logpdf(z)
            z_logcdf = norm.logcdf(z)

            hess_likelihood_x = self._compute_hess_likelihood([z, z_logpdf, z_logcdf])
            hess_likelihood_x_dx[:f.shape[0], :f.shape[0]] = hess_likelihood_x

        # Hessian of virtual derivative likelihood
        if df.shape[0] > 0:
            m = 1. / nu * df
            m_logpdf = norm.logpdf(m)
            m_logcdf = norm.logcdf(m)
            hess_likelihood_dx = - 1. / (nu ** 2) * np.diag(
                np.exp(2 * m_logpdf - 2 * m_logcdf) + df * np.exp(m_logpdf - m_logcdf))
            hess_likelihood_x_dx[f.shape[0]:, f.shape[0]:] = hess_likelihood_dx

        return hess_likelihood_x_dx

    def _compute_posterior(self):

        converged = False
        nu = self.nu
        num_tries = 3
        try_no = 0

        # using Newton-Raphson, approximate f_MAP
        while not converged and try_no < num_tries:

            # randomly initialise f_map
            f_map = self.random_state.uniform(0., 1., self.datapoints.shape[0])
            df_map = self.random_state.uniform(0., 1., self.datapoints_deriv.shape[0]*self.num_objectives)
            f_df_map = np.hstack((f_map, df_map))

            for lapl_iter in range(100+50*(try_no+1)):

                # gradient of pairwise likelihood
                f_winner = np.array([f_map[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
                f_loser = np.array([f_map[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
                z = (f_winner - f_loser) / (np.sqrt(2) * self.std_noise)
                z_logpdf = norm.logpdf(z)
                z_logcdf = norm.logcdf(z)

                # compute b1
                if len(self.datapoints) > 0:
                    h_j = np.array([np.array(self.comparisons[:, 0] == j, dtype=int) - np.array(self.comparisons[:, 1] == j,
                                    dtype=int) for j in range(self.datapoints.shape[0])])
                    b1 = np.sum(h_j * np.exp(z_logpdf - z_logcdf), axis=1) / (np.sqrt(2) * self.std_noise)
                else:
                    b1 = []

                if self.datapoints_deriv.shape[0] > 0:
                    m = 1./nu * df_map
                    m_logpdf = norm.logpdf(m)
                    m_logcdf = norm.logcdf(m)
                    b2 = np.exp(-np.log(nu) + m_logpdf - m_logcdf)
                else:
                    b2 = []

                # stack together to b
                b = np.hstack((b1, b2))

                # compute gradient g
                g = - np.dot(self.cov_mat_inv, (f_df_map - self.prior_joint_mean(self.datapoints, self.datapoints_deriv)))
                g += b

                # compute hessian of posterior
                hess_likelihood_x_dx = self._compute_hess_likelihood_x_dx(f_df_map, nu)
                hess_posterior = - self.cov_mat_inv + hess_likelihood_x_dx
                hess_posterior_inv = np.linalg.inv(hess_posterior)

                update = np.dot(hess_posterior_inv, g)
                f_df_map -= update

                f_map = f_df_map[:self.datapoints.shape[0]]
                df_map = f_df_map[self.datapoints.shape[0]:]

                # stop criterion
                if np.linalg.norm(update) < 0.00001:
                    converged = True
                    break

            if not converged:
                print("Did not converge with {}".format(nu))
                try_no += 1
                nu /= 10
                if try_no == num_tries:
                    print("stopping the search.")
                    print(np.linalg.norm(update))

        # update lambda
        self.lambda_mat = hess_likelihood_x_dx
        self.lambda_mat_inv = np.linalg.inv(self.lambda_mat)

        return f_df_map
