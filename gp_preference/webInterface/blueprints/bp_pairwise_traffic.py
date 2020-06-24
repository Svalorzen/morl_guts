import numpy as np
import time
from flask import Blueprint
from flask import render_template, request, redirect, url_for
import sys
sys.path.insert(0, '../')
from python_backend import specs_traffic, utils_users, data_traffic
sys.path.insert(0, '../../')
from gaussian_process import GPPairwise
from acquisition_function import DiscreteAcquirer

bp_pairwise_traffic = Blueprint('bp_pairwise_traffic', __name__)


# --- pairwise experiment --

@bp_pairwise_traffic.route('/start_pairwise_traffic/<username>', methods=['POST', 'GET'])
def start(username):

    # register start time for this user
    utils_users.get_experiment_start_time(username, 'pairwise')

    # initialise acquirer
    acquirer = DiscreteAcquirer(input_domain=data_traffic.get_traffic_results(), query_type='clustering',
                                seed=specs_traffic.SEED)

    # get the starting points from the acquirer
    item1, item2 = -1*np.array(acquirer.get_start_points())

    # get the objective names
    obj_names = data_traffic.get_objective_names()
    obj_abbrev = data_traffic.get_objective_abbrev()

    return render_template("query_pairwise_traffic.html", username=username, item1=item1, item2=item2, side_clicked=-1,
                           obj_names=obj_names, obj_abbrev=obj_abbrev)


@bp_pairwise_traffic.route('/choose_pairwise_traffic', methods=['POST'])
def choose_pairwise():

    # get current user
    username = request.form['username']

    # get the comparison
    winner_item = -np.fromstring(request.form['winner'][:-2], sep=',')
    loser_item = -np.fromstring(request.form['loser'][:-2], sep=',')

    # add comparison to user's data and save
    dataset_user = utils_users.get_gp_dataset(username, 'pairwise', num_objectives=specs_traffic.NUM_OBJECTIVES)
    dataset_user.add_single_comparison(winner_item, loser_item)
    utils_users.update_gp_dataset(username, dataset_user, 'pairwise')

    # find out whether user clicked left or right button (0=left, 1=right)
    item_clicked = request.form['item-clicked']

    button_type = request.form['buttonType']

    # display new traffic
    return redirect(url_for('.continue_pairwise', username=username, side_clicked=item_clicked, button_type=button_type))


@bp_pairwise_traffic.route('/continue_pairwise_traffic/<username>_<side_clicked>_<button_type>', methods=['POST', 'GET'])
def continue_pairwise(username, side_clicked, button_type):

    # get the dataset for this user
    dataset_user = utils_users.get_gp_dataset(username, 'pairwise', num_objectives=specs_traffic.NUM_OBJECTIVES)

    # initialise the acquirer which picks new datapoints
    acquirer = DiscreteAcquirer(input_domain=data_traffic.get_traffic_results(), query_type='pairwise',
                                seed=specs_traffic.SEED)

    # add collected datapoints to acquirer
    acquirer.history = dataset_user.datapoints

    # add virtual comparisons in first few queries
    for i in range(np.min([dataset_user.datapoints.shape[0], 6])):
        dataset_user.add_single_comparison(dataset_user.datapoints[i], data_traffic.get_traffic_min())
        dataset_user.add_single_comparison(data_traffic.get_traffic_max(), dataset_user.datapoints[i])

    # add linear prior in first few queries
    if acquirer.history.shape[0] < 6:
        prior_mean_type = 'linear'
    else:
        prior_mean_type = 'zero'

    # intialise the GP
    gp = GPPairwise(num_objectives=specs_traffic.NUM_OBJECTIVES, seed=specs_traffic.SEED, prior_mean_type=prior_mean_type)

    # add collected datapoints to GP
    gp.update(dataset_user)

    # get the best job so far
    job_best_idx = dataset_user.comparisons[-1, 0]
    job_best = dataset_user.datapoints[job_best_idx]

    # let acquirer pick new point
    job_new = acquirer.get_next_point(gp, dataset_user)

    # sort according to what user did last round
    if side_clicked == "1":
        item1 = job_best
        item2 = job_new
    else:
        item1 = job_new
        item2 = job_best

    # get the names of the objectives
    obj_names = data_traffic.get_objective_names()
    obj_abbrev = data_traffic.get_objective_abbrev()

    if button_type == 'next':
        return render_template("query_pairwise_traffic.html", username=username, item1=-item1, item2=-item2,
                               side_clicked=side_clicked, obj_names=obj_names, obj_abbrev=obj_abbrev)
    elif button_type == 'end':
        # save end time
        utils_users.save_experiment_end_time(username, 'pairwise')
        # register that this experiment was done
        utils_users.update_experiment_status(username=username, query_type='pairwise')
        return redirect('start_experiment/{}'.format(username))
    else:
        raise NotImplementedError('Button type unknown.')
