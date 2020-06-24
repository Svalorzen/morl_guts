import re
import numpy as np
from os import environ
from flask import Flask, render_template, request, redirect
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from python_backend import data_traffic, utils_users
# import the blueprints for the tutorials
from blueprints.bp_pairwise_tutorial_traffic import bp_pairwise_tutorial
from blueprints.bp_ranking_tutorial_traffic import bp_ranking_tutorial
# import the blueprints for the experiment
from blueprints.bp_pairwise_traffic import bp_pairwise_traffic
from blueprints.bp_ranking_traffic import bp_ranking_traffic

# create application
app = Flask(__name__)

# register the blueprints for the tutorial
app.register_blueprint(bp_pairwise_tutorial)
app.register_blueprint(bp_ranking_tutorial)

# register the blueprints for the different experiments
app.register_blueprint(bp_pairwise_traffic)
app.register_blueprint(bp_ranking_traffic)


@app.route("/")
def main():
    return render_template('index_traffic.html')


# -- start page  --

@app.route('/register_user', methods=['POST'])
def register_user():
    username = request.form['username']

    # if the input is not OK, set username to None and return
    if not re.match('^\w+$', username):
        return render_template('index_traffic.html', username='invalid')

    # check if user exists already and completed the survey
    user_status = utils_users.get_user_status(username)
    if len(user_status['survey results']) > 0:
        return render_template('index_traffic.html', username='existing')

    utils_users.register_user(username)

    return render_template("persona_tutorial_traffic.html", username=username)

# -- tutorial --


@app.route('/persona_tutorial', methods=['POST'])
def persona_tutorial():

    # get the username
    username = request.form['username']

    # go to the persona description page (using the username)
    return render_template("persona_tutorial_traffic.html", username=username)


@app.route('/start_tutorial', methods=['POST'])
def start_tutorial():

    # get the username
    username = request.form['username']

    # get the user's status on the tutorials
    tutorial_status = utils_users.get_tutorial_status(username)

    # go to the persona description page (using the username)
    return render_template("navi_tutorial_traffic.html", username=username, pairwise_fin=tutorial_status[0],
                           ranking_fin=tutorial_status[1], clustering_fin=tutorial_status[2])


@app.route('/finish_tutorial', methods=['POST'])
def finish_tutorial():

    # get the username
    username = request.form['username']

    return render_template('index.html', username=username, tutorial_completed=True)


# -- experiment --

@app.route('/reroute_to_experiment', methods=['POST'])
def reroute_to_experiment():
    # get the username
    username = request.form['username']
    # get an example for the traffic results
    traffic_example = -data_traffic.get_example_traffic()
    # get the names and abbreviations of the objectives
    obj_names = data_traffic.get_objective_names()
    obj_abbrev = data_traffic.get_objective_abbrev()
    # go to the persona description page (using the username)
    return render_template("persona_traffic.html", username=username, traffic_example=traffic_example,
                           obj_names=obj_names, obj_abbrev=obj_abbrev)


@app.route('/start_experiment/<username>')
def start_experiment(username):

    # decide which query type the user has to do now (pairwise, clustering, ranking)
    query_type = utils_users.get_next_experiment_type(username)

    if query_type == 'clustering':
        query_type = None

    if query_type is not None:
        return render_template("navi_experiment_traffic.html", username=username, query_type=query_type)
    else:
        return redirect('start_survey_traffic/{}'.format(username))


# -- survey --

@app.route('/start_survey_traffic/<username>')
def start_survey(username):

    # get the resulting traffic
    user_status = utils_users.get_user_status(username)

    win_idx_pair = user_status['dataset pairwise'].comparisons[-1, 0]
    win_pair = -1*user_status['dataset pairwise'].datapoints[win_idx_pair]

    win_idx_rank = user_status['logs ranking'][-1][0]
    win_rank = -1*user_status['dataset ranking'].datapoints[win_idx_rank]

    # shuffle the results
    order = np.random.permutation([0, 1])
    winner_ids = np.array([win_idx_pair, win_idx_rank])[order]
    winners = np.array([win_pair, win_rank])[order]

    # get the object abbreviations
    obj_names = data_traffic.get_objective_names()
    obj_abbrev = data_traffic.get_objective_abbrev()

    return render_template('survey_traffic.html', username=username, winners=winners, winner_ids=winner_ids, obj_names=obj_names, obj_abbrev=obj_abbrev)


@app.route('/submit_survey', methods=['POST'])
def submit_survey():

    # get the user name
    username = request.form['username']

    survey_result = {
        'outcome order': request.form['outcome-ranking'],
        'understanding': [request.form['understand-pairwise'], request.form['understand-ranking']],
        'preference': request.form['preference-ranking'],
        'effort': [request.form['pairwise-effort'], request.form['ranking-effort']],
        'comment': request.form['comment']
    }

    utils_users.save_survey_result(username, survey_result)

    return render_template('the_end_traffic.html', username=username)


# --- run ---

if __name__ == "__main__":
    # app.run(host='0.0.0.0', debug=True, port=environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=80)
