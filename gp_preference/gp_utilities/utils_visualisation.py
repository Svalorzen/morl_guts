import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!


class Visualiser:

    def __init__(self, x_dim, num_queries, scalariser, acquirer):

        self.x_dim = x_dim
        self.x_min = np.zeros(x_dim)
        self.x_max = np.ones(x_dim)
        self.num_queries = num_queries
        self.idx_query = 1
        self.scalariser = scalariser
        self.acquirer = acquirer

        plt.figure()

    def update_graph(self, gp, ccs, x_curr_max, f_curr_max):

        # self.visualise_results_2d(gp, ccs_cache, x_curr_max, f_curr_max)
        self.visualise_results_2d(gp, ccs, x_curr_max, f_curr_max)

        if self.idx_query == self.num_queries:
            plt.show()

        self.idx_query += 1

    def visualise_results_2d(self, gp, ccs, x_curr_max, f_curr_max):

        # get the data we observed in the GP so far
        x_data = gp.datapoints if gp.datapoints is not None else np.zeros((1, self.x_dim))
        f_data = gp.utility_vals if gp.datapoints is not None else np.zeros(1)

        # construct meshgrid over whole [0,1]x[0,1] space
        n = 50
        meshs = np.meshgrid(*[np.linspace(self.x_min[i], self.x_max[i], n) for i in range(self.x_dim)])
        x_mesh = np.vstack([meshs[i].flatten() for i in range(self.x_dim)]).T

        # evaluate the true scalarisation function and make predictions using the GP
        f_mesh_true = self.scalariser.get_preference(x_mesh, add_noise=False).reshape(meshs[0].shape)

        # now scale so that the y-values are always between 0 and 1
        if f_curr_max is not None:
            f_curr_max_s = scale_unit(f_curr_max, f_mesh_true)

        # --- plot the thing reduced to CCS

        plot_width = 5
        plot_height = (self.num_queries-1)/5 + 1

        plt.subplot(plot_height, plot_width, self.idx_query)

        x_ccs = ccs

        f_ccs_true = self.scalariser.get_preference(x_ccs, add_noise=False)
        f_ccs_approx_mean, f_ccs_approx_vars = gp.get_predictive_params(x_ccs, True)
        # ei_ccs = acquirer.get_expected_improvement(x_ccs, gp, np.empty((0, 1)))

        # before plotting, normalize everything to [0, 1]
        f_data_s = scale_unit(f_data, f_ccs_approx_mean)
        f_ccs_true_s = scale_unit(f_ccs_true)
        if f_curr_max is not None:
            f_curr_max_s = scale_unit(f_curr_max, f_ccs_true)
        f_ccs_approx_mean_s = scale_unit(f_ccs_approx_mean)
        # ei_ccs_s = scale_unit(ei_ccs)
        # I get a weird 'read-only' error for this vector so I do this to circumvent it:
        f_ccs_approx_vars = np.array(f_ccs_approx_vars)
        f_ccs_approx_vars *= 0.1

        # plt.plot(x_ccs[:, 0], ei_ccs_s, 'g-', label='EI')
        plt.plot(x_data[:, 0], f_data_s[:x_data.shape[0]], 'b*')
        if f_curr_max is not None:
            plt.plot(x_curr_max[0], f_curr_max_s, 'ro')
        plt.plot(x_ccs[:, 0], f_ccs_true_s,  'b', label='true')
        plt.plot(x_ccs[:, 0], f_ccs_approx_mean_s, 'r', label='approx')
        # plt.plot(x_ccs[:, 0], f_ccs_approx_vars, 'm', label='variance')
        f_ccs_approx_vars = np.array(f_ccs_approx_vars)
        f_ccs_approx_vars[f_ccs_approx_vars < 0] = 0
        plt.fill_between(x_ccs[:, 0], f_ccs_approx_mean_s - np.sqrt(f_ccs_approx_vars), f_ccs_approx_mean_s + np.sqrt(f_ccs_approx_vars),
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='estimate (var)')

        plt.ylim([-0.1, 1.1])
        plt.xlim([0, 1])
        plt.xlabel('x_0')

    def visualise_results_3d(self, gp, ccs, x_curr_max, f_curr_max):

        # get the data we observed in the GP so far
        x_data = gp.datapoints if gp.datapoints is not None else np.zeros((1, self.x_dim))
        f_data = gp.utility_vals if gp.datapoints is not None else np.zeros(1)

        # construct meshgrid over whole [0,1]x[0,1] space
        n = 50
        meshs = np.meshgrid(*[np.linspace(self.x_min[i], self.x_max[i], n) for i in range(self.x_dim)])
        x_mesh = np.vstack([meshs[i].flatten() for i in range(self.x_dim)]).T

        # evaluate the true scalarisation function and make predictions using the GP
        f_mesh_true = self.scalariser.get_preference(x_mesh, scale=True, add_noise=False).reshape(meshs[0].shape)
        f_mesh_approx = gp.get_predictive_params(x_mesh, pointwise=True)[0].reshape(meshs[0].shape)

        # now scale so that the y-values are always between 0 and 1
        if f_curr_max is not None:
            f_curr_max_s = scale_unit(f_curr_max, f_mesh_true)

        # now scale so that the y-values are always between 0 and 1
        f_data_s = scale_unit(f_data, f_mesh_approx)
        f_mesh_approx_s = scale_unit(f_mesh_approx)
        f_mesh_true_s = scale_unit(f_mesh_true)

        height_plot = 2
        width_plot = self.num_queries + 1

        # plot the CCS
        plt.subplot(height_plot, width_plot, 1)
        plt.plot(ccs.datapoints[:, 0], ccs.datapoints[:, 1], 'r*')
        plt.title('CCS')

        # pick the current subplot
        ax = plt.subplot(height_plot, width_plot, self.idx_query+1, projection='3d')

        # plot the true datapoints
        ax.scatter(x_data[:, 0], x_data[:, 1], f_data_s, c='b', label='datapoints')
        # ax.scatter(x_curr_max[0], x_curr_max[1], f_curr_max, c='r', label='datapoints')

        # plot the true function values
        ax.plot_wireframe(meshs[0], meshs[1], f_mesh_true_s, color='blue', rstride=10, cstride=10, label='true')

        # plot the GP
        ax.plot_wireframe(meshs[0], meshs[1], f_mesh_approx_s, color='red', rstride=10, cstride=10, label='GP')

        if self.idx_query == self.num_queries:
            plt.legend(loc=2)

        # --- plot the thing reduced to CCS

        # plot the datapoints
        plt.subplot(height_plot, width_plot, width_plot + 1)
        plt.plot(x_data[:, 0], x_data[:, 1], 'r*')
        plt.title('datapoints')

        plt.subplot(height_plot, width_plot, self.idx_query + width_plot + 1)

        x_ccs = np.empty((0, self.x_dim))
        n = 100
        for a in ccs.simplices:
            l = np.linspace(0, 1, n)[:, np.newaxis]
            x_ccs = np.vstack((x_ccs, (1.-l)*a[0] + l*a[1]))

        f_ccs_true = self.scalariser.get_preference(x_ccs, add_noise=False)
        f_ccs_approx_mean, f_ccs_approx_vars = gp.get_predictive_params(x_ccs, pointwise=True)
        ei_ccs = self.acquirer._compute_expected_improvement(x_ccs, gp)

        # before plotting, normalize everything to [0, 1]
        f_ccs_true_s = scale_unit(f_ccs_true)
        # f_curr_max_s = scale_unit(f_curr_max, f_ccs_true)
        f_ccs_approx_mean_s = scale_unit(f_ccs_approx_mean)
        ei_ccs_s = scale_unit(ei_ccs)
        f_ccs_approx_vars = np.array(f_ccs_approx_vars)
        f_ccs_approx_vars *= 0.5

        plt.plot(x_ccs[:, 0], ei_ccs_s, 'g-', label='EI')
        plt.plot(x_data[:, 0], f_data_s, 'b*')
        # plt.plot(x_curr_max[0], f_curr_max, 'ro')
        plt.plot(x_ccs[:, 0], f_ccs_true_s,  'b', label='true')
        plt.plot(x_ccs[:, 0], f_ccs_approx_mean_s, 'r', label='approx')
        plt.plot(x_ccs[:, 0], f_ccs_approx_vars, 'm', label='variance')
        f_ccs_approx_vars = np.array(f_ccs_approx_vars)
        f_ccs_approx_vars[f_ccs_approx_vars < 0] = 0
        plt.fill_between(x_ccs[:, 0], f_ccs_approx_mean - np.sqrt(f_ccs_approx_vars), f_ccs_approx_mean + np.sqrt(f_ccs_approx_vars),
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='estimate (var)')

        plt.ylim([-0.1, 1.1])

        if self.idx_query == 1:
            plt.legend(loc=2)


def scale_unit(x, y=None):

    if y is None:
        y_min = np.min(x)
        y_max = np.max(x)
    else:
        y_min = np.min(y)
        y_max = np.max(y)

    if y_min != y_max:
        x_scaled = (x - y_min) / (y_max - y_min)

    elif 1 > y_max > 0:
        x_scaled = x

    else:
        x_scaled = np.zeros(x.shape)

    return x_scaled