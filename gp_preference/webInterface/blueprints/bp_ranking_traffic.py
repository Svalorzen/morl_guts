import numpy as np
from flask import render_template, request, redirect, url_for, Blueprint
import sys
import time
sys.path.insert(0, '../')
from python_backend import utils_users, specs_traffic, data_traffic
sys.path.insert(0, '../../')
from gaussian_process import GPPairwise
from acquisition_function import DiscreteAcquirer

bp_ranking_traffic = Blueprint('bp_ranking_traffic', __name__)


@bp_ranking_traffic.route('/start_ranking_traffic/<username>', methods=['POST', 'GET'])
def start(username):

    # get the start time for this user
    utils_users.get_experiment_start_time(username, 'ranking')

    # get the dataset for this user
    user_dataset = utils_users.get_gp_dataset(username, 'ranking', num_objectives=specs_traffic.NUM_OBJECTIVES)

    # initialise acquirer
    acquirer = DiscreteAcquirer(input_domain=data_traffic.get_traffic_results(), query_type='clustering',
                                seed=specs_traffic.SEED)
    
    # if no data has been collected yet, we only display two starting traffic
    if user_dataset.comparisons.shape[0] == 0:

        # delete any datapoint in the user's dataset (in case experiment was aborted)
        user_dataset.datapoints = np.empty((0, specs_traffic.NUM_OBJECTIVES))

        # get the starting points from the acquirer
        item1, item2 = acquirer.get_start_points()

        # add traffic to dataset of user
        item1_ID = user_dataset._add_single_datapoint(item1)
        item2_ID = user_dataset._add_single_datapoint(item2)

        # save dataset
        utils_users.update_gp_dataset(username, user_dataset, 'ranking')

        # put traffic we want to display in the respective lists
        traffic_unranked = [item1, item2]
        IDs_unranked = [item1_ID, item2_ID]
        traffic_ranked = []
        IDs_ranked = []

    # otherwise, we show the previous ranking and pick a new point according to that
    else:

        # add collected datapoints to acquirer
        acquirer.history = user_dataset.datapoints

        # add virtual comparisons in first few queries
        for i in range(np.min([user_dataset.datapoints.shape[0], 6])):
            user_dataset.add_single_comparison(user_dataset.datapoints[i], data_traffic.get_traffic_min())
            user_dataset.add_single_comparison(data_traffic.get_traffic_max(), user_dataset.datapoints[i])

        # add linear prior in first few queries
        if acquirer.history.shape[0] < 6:
            prior_mean_type = 'linear'
        else:
            prior_mean_type = 'zero'

        print("comparisons after adding stuff", user_dataset.comparisons)

        # intialise the GP
        gp = GPPairwise(num_objectives=specs_traffic.NUM_OBJECTIVES, seed=specs_traffic.SEED, prior_mean_type=prior_mean_type)

        # add collected datapoints to GP
        gp.update(user_dataset)

        # let acquirer pick new point
        job_new = acquirer.get_next_point(gp, user_dataset)

        # add that point to the dataset and save
        job_new_ID = user_dataset._add_single_datapoint(job_new)
        utils_users.update_gp_dataset(username, user_dataset, 'ranking')

        # put into list of traffic that need to be ranked
        traffic_unranked = [job_new]
        IDs_unranked = [job_new_ID]

        # get ranking so far
        IDs_ranked = utils_users.get_ranking(username)
        # get the job information from that ranking and convert to dictionaries
        traffic_ranked = user_dataset.datapoints[IDs_ranked]

    # get names of objectives
    obj_names = data_traffic.get_objective_names()
    obj_abbrev = data_traffic.get_objective_abbrev()

    return render_template("query_ranking_traffic.html", username=username, traffic_unranked=-1*np.array(traffic_unranked),
                           traffic_ranked=-1*np.array(traffic_ranked), IDs_ranked=IDs_ranked, IDs_unranked=IDs_unranked,
                           obj_names=obj_names, obj_abbrev=obj_abbrev)


@bp_ranking_traffic.route('/submit_ranking_traffic', methods=['POST'])
def submit_ranking():

    # get the username and their dataset
    username = request.form['username']
    user_dataset = utils_users.get_gp_dataset(username, 'ranking', num_objectives=specs_traffic.NUM_OBJECTIVES)

    # get the ranking the user submitted
    ranking = np.fromstring(request.form['rankingResult'], sep=',', dtype=np.int)
    # save it
    utils_users.save_ranking(username, ranking)

    # get the actual traffic (ranking returned the indiced in the dataset)
    traffic_ranked = user_dataset.datapoints[ranking]

    # add the ranking to the dataset
    user_dataset.add_ranked_preferences(traffic_ranked)

    print("comparisons from curr ranking", user_dataset.comparisons)

    # save
    utils_users.update_gp_dataset(username, user_dataset, 'ranking')

    # get the start time for this user
    utils_users.get_experiment_start_time(username, 'ranking')

    button_type = request.form['buttonType']

    if button_type == 'next':
        return redirect(url_for('.start', username=username))
    elif button_type == 'end':
        # save end time
        utils_users.save_experiment_end_time(username, 'ranking')
        # register that this experiment was done
        utils_users.update_experiment_status(username=username, query_type='ranking')
        return redirect('start_experiment/{}'.format(username))
    else:
        raise NotImplementedError('Button type not defined')
