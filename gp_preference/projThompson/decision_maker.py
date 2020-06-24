import copy
import numpy as np

from gaussian_process import GPPairwise

from dataset import DatasetPairwise
from gp_utilities.utils_user import UserPreference

class DecisionMaker:

    def __init__(self, num_objectives, seed, utility_function=None, user_std=0.1, temp_linear_prior=False, add_virtual_comp=False, add_virt_comp_global=False, keep_set_small=False, thresh_dist=0.001):

        self.random_state = np.random.RandomState(seed)
        if utility_function is None:
            user = UserPreference(num_objectives, user_std, seed)
            self.utility_function = user.get_preference
        else:
            self.utility_function = lambda x, add_noise: utility_function(x) + int(add_noise) * self.random_state.normal(0, user_std)
        self.dataset = DatasetPairwise(num_objectives)
        self.gp = GPPairwise(num_objectives, kernel_width=0.45, std_noise=user_std, seed=seed)
        self.temp_linear_prior = temp_linear_prior
        self.add_virtual_comp = add_virtual_comp
        self.min_point = None
        self.max_point = None
        self.add_virt_comp_global = add_virt_comp_global
        self.keep_set_small = keep_set_small
        self.thresh_dist = thresh_dist


    def true_utility(self, vect):
        return self.utility_function(vect, add_noise=False)

    def set_prior(self):
        if self.temp_linear_prior:
            if self.dataset.comparisons.shape[0] < 10:
                self.gp.prior_mean_type = 'linear'
            else:
                self.gp.prior_mean_type = 'zero'

    def noisy_compare(self, vect1, vect2, dont_update=False):
        """ adds comparison to dataset, returns boolean vect1>vect """
        if self.keep_set_small and self.dataset.datapoints.shape[0] > 0:
            dist1 = np.linalg.norm(self.dataset.datapoints - vect1, axis=1)
            if np.min(dist1) < self.thresh_dist:
                vect1 = self.dataset.datapoints[np.argmin(dist1)]
            dist2 = np.linalg.norm(self.dataset.datapoints - vect2, axis=1)
            if np.min(dist2) < self.thresh_dist:
                vect2 = self.dataset.datapoints[np.argmin(dist1)]
        utl1 = self.utility_function(vect1, add_noise=True)
        utl2 = self.utility_function(vect2, add_noise=True)
        if utl1 > utl2:
            self.dataset.add_single_comparison(vect1, vect2)
        else:
            self.dataset.add_single_comparison(vect2, vect1)
        if not dont_update:
            self.update_gp(self.dataset)
        return utl1 > utl2

    def sample(self, sample_points):
        """ returns a sample of the GP utility at sample_points """

        # if requested, add virtual comparisons to nadir and utopian point
        if self.add_virtual_comp:
            self.virtual_comp(sample_points)
        if self.add_virt_comp_global:
            self.virtual_comp_global(sample_points)

        return self.gp.sample(sample_points)

    def update_gp(self, dataset):
        self.set_prior()
        self.gp.update(dataset)

    def virtual_comp(self, sample_points):

        # put list of PF vectors into matrix
        sample_points = np.vstack(sample_points)

        # check if the pareto front is more than just a single point
        if sample_points.shape[0] == 1:

            # make sure the correct data is used (we might've added virtual comparisons last round)
            if self.dataset.datapoints.shape[0] > 0:
                self.update_gp(self.dataset)

        else:

            # get utopian and nadir points
            utopian_point = np.max(sample_points, axis=0)
            nadir_point = np.min(sample_points, axis=0)

            # copy current dataset
            dataset_copy = copy.deepcopy(self.dataset)

            # add virtual comparisons
            for i in range(sample_points.shape[0]):
                vect = sample_points[i]
                dataset_copy.add_single_comparison(utopian_point, vect)
                dataset_copy.add_single_comparison(vect, nadir_point)

            # update the GP using this dataset (note: the GP forgets everything it knew before)
            self.update_gp(dataset_copy)

    def virtual_comp_global(self, sample_points):

        # put list of PF vectors into matrix
        sample_points = np.vstack(sample_points)

        if self.max_point is None:
            self.max_point = np.max(sample_points, axis=0)
        else:
            stacked_points = np.vstack((sample_points, self.max_point))
            self.max_point = np.max(stacked_points, axis=0)
        if self.min_point is None:
            self.min_point = np.min(sample_points, axis=0)
        else:
            stacked_points = np.vstack((sample_points, self.min_point))
            self.min_point = np.min(stacked_points, axis=0)

        # copy current dataset
        dataset_copy = copy.deepcopy(self.dataset)

        # add virtual comparisons
        for i in range(sample_points.shape[0]):
            vect = sample_points[i]
            if np.sum(np.abs(vect-self.max_point)) != 0:
                dataset_copy.add_single_comparison(self.max_point, vect)
            if np.sum(np.abs(vect-self.min_point)) != 0:
                dataset_copy.add_single_comparison(vect, self.min_point)

        # update the GP using this dataset (note: the GP forgets everything it knew before)
        if dataset_copy.comparisons.shape[0] > 0:
            self.update_gp(dataset_copy)


