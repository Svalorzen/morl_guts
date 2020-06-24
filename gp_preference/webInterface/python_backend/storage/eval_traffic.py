import pickle as pkl
import time

PATH_TRAFFIC1 = './users_traffic1.p'
PATH_TRAFFIC2 = './users_traffic2.p'

traffic1 = pkl.load(open(PATH_TRAFFIC1, 'rb'))
traffic2 = pkl.load(open(PATH_TRAFFIC2, 'rb'))

print(traffic1.keys())
print(traffic2.keys())
print(traffic1['sjoerd'].keys())
print('')

results = [traffic1['Freek'], traffic1['marcel'], traffic1['stef'], traffic1['sjoerd'],
           traffic2['stef'], traffic2['sjoerd2']]

for r in results:
    if r['dataset pairwise'] is not None:
        print(r['name'])
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r['start time ranking'])))
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r['start time pairwise'])))
        # print(r['dataset pairwise'].datapoints.shape)
        # print(r['dataset ranking'].datapoints.shape)
        print(r['dataset pairwise'].comparisons)
        print(r['logs ranking'])
        # print(r)
        # print(r['survey results'])
        # print('')
        # print(-1*r['dataset pairwise'].datapoints[r['dataset pairwise'].comparisons[-1][0]])
        # print(-1*r['dataset pairwise'].datapoints[r['dataset pairwise'].comparisons[-1][1]])
        # print(-1*r['dataset ranking'].datapoints[r['logs ranking'][-1][0]])
        print('')

# ALL_JOBS = utils_jobs.get_jobs()
#
# # (1) utility that was reached per query
# utl_per_step_p = np.empty((0, 0))
# utl_per_step_c = np.empty((0, 0))
# utl_per_step_r = np.empty((0, 0))
#
# # (2) number of datapoints for users that completed everything
# queries_answered = np.empty((0, 3))
#
# # (3) how much effort it was
# effort = np.zeros((3, 2))
#
# # (4) which of the methods the user preferred
# preferences = np.zeros((3, 3))
#
# # (5) which outcome they think was best vs what was actually best
# outcome_rank_true = np.zeros((3, 3))
# outcome_rank_felt = np.zeros((3, 3))
#
# # (1a) utility that was reached per query - for those that understood the task
# utl_per_step_p_CORR = np.empty((0, 0))
# utl_per_step_c_CORR = np.empty((0, 0))
# utl_per_step_r_CORR = np.empty((0, 0))
# queries_answered_CORR = np.empty((0, 3))
#
# # (6) how much effort it was
# understanding = np.zeros((3, 3))
#
#
# def get_utility_per_step(user, utl_p, utl_c, utl_r):
#
#     # pairwise
#     p_indices = user['dataset pairwise'].comparisons[:, 0]
#     p_jobs = ALL_JOBS[p_indices]
#     p_utilities = utility_func_jobs.utility(p_jobs)
#     if utl_p.shape[1] < len(p_utilities):
#         if utl_p.shape[1] > 0:
#             padding_p = np.zeros((utl_p.shape[0], len(p_utilities) - utl_p.shape[1])) + utl_p[:, -1][:, np.newaxis]
#         else:
#             padding_p = np.zeros((utl_p.shape[0], len(p_utilities)))
#         utl_p = np.hstack((utl_p, padding_p))
#     utl_p = np.vstack((utl_p, np.hstack((p_utilities, np.zeros(utl_p.shape[1] - len(p_utilities)) + p_utilities[-1]))))
#
#     # clustering
#     c_indices = [log[0][0] for log in user['logs clustering']]
#     c_jobs = ALL_JOBS[c_indices]
#     c_utilities = utility_func_jobs.utility(c_jobs)
#     if utl_c.shape[1] < len(c_utilities):
#         if utl_c.shape[1] > 0:
#             padding_c = np.zeros((utl_c.shape[0], len(c_utilities) - utl_c.shape[1])) + utl_c[:, -1][:, np.newaxis]
#         else:
#             padding_c = np.zeros((utl_c.shape[0], len(c_utilities)))
#         utl_c = np.hstack((utl_c, padding_c))
#     utl_c = np.vstack((utl_c, np.hstack((c_utilities, np.zeros(utl_c.shape[1] - len(c_utilities)) + c_utilities[-1]))))
#
#     # ranking
#     r_indices = [log[0] for log in user['logs ranking']]
#     r_jobs = ALL_JOBS[r_indices]
#     r_utilities = utility_func_jobs.utility(r_jobs)
#     if utl_r.shape[1] < len(r_utilities):
#         if utl_r.shape[1] > 0:
#             padding_r = np.zeros((utl_r.shape[0], len(r_utilities) - utl_r.shape[1])) + utl_r[:, -1][:, np.newaxis]
#         else:
#             padding_r = np.zeros((utl_r.shape[0], len(r_utilities)))
#         utl_r = np.hstack((utl_r, padding_r))
#     utl_r = np.vstack((utl_r, np.hstack((r_utilities, np.zeros(utl_r.shape[1] - len(r_utilities)) + r_utilities[-1]))))
#
#     return utl_p, utl_c, utl_r
#
#
# def get_true_winners(user_status):
#     win_idx_pair = user_status['dataset pairwise'].comparisons[-1, 0]
#     win_pair = user_status['dataset pairwise'].datapoints[win_idx_pair]
#
#     win_idx_clust = user_status['logs clustering'][-1][0][0]
#     win_clust = user_status['dataset clustering'].datapoints[win_idx_clust]
#
#     win_idx_rank = user_status['logs ranking'][-1][0]
#     win_rank = user_status['dataset ranking'].datapoints[win_idx_rank]
#
#     winners = [win_pair, win_clust, win_rank]
#
#     return winners
#
#
# PATH_RESULTS = 'users_mturk.p'
#
# users = pkl.load(open(PATH_RESULTS, 'rb'))
# print('num results:', len(users), '\n')
#
# successful_users = []
#
# for k in range(len(users)):
#
#     key = list(users.keys())[k]
#     user = users[key]
#
#     if len(user['survey results']) > 0:
#
#         successful_users.append(user['name'])
#
#         # (1) utility reached per query
#         utl_per_step_p, utl_per_step_c, utl_per_step_r = get_utility_per_step(user, utl_per_step_p, utl_per_step_c, utl_per_step_r)
#
#         # (2) how many queries were answered
#         queries_answered = np.vstack((queries_answered, np.zeros(3)))
#         curr_idx = queries_answered.shape[0]
#         queries_answered[curr_idx - 1, 0] = user['dataset pairwise'].datapoints.shape[0]
#         queries_answered[curr_idx - 1, 1] = user['dataset clustering'].datapoints.shape[0]
#         queries_answered[curr_idx - 1, 2] = user['dataset ranking'].datapoints.shape[0]
#
#         # (3) how much effort they thought it was
#         user_effort = user['survey results']['effort']
#         for i in range(3):
#             if user_effort[i] == 'OK':
#                 effort[i, 0] += 1
#             else:
#                 effort[i, 1] += 1
#
#         # (4) which of the methods they prefer most/least
#         user_prefs = user['survey results']['preference'].split(',')
#         print(user_prefs[0])
#         preferences[0, user_prefs.index('prefer-pairwise')] += 1
#         preferences[1, user_prefs.index('prefer-clustering')] += 1
#         preferences[2, user_prefs.index('prefer-ranking')] += 1
#
#         # (5) which outcome they think is best vs what is actually best
#         true_winners = utility_func_jobs.utility(np.vstack(get_true_winners(user)))
#         user_winner = user['survey results']['outcome order'].split(',')
#         user_winner[user_winner.index('pairwise')] = 0
#         user_winner[user_winner.index('clustering')] = 1
#         user_winner[user_winner.index('ranking')] = 2
#
#         slack = 0
#         if true_winners[user_winner[0]]+slack*2 >= true_winners[user_winner[1]]+slack >= true_winners[user_winner[2]] and user['survey results']['distraction'] == 'NO':
#             utl_per_step_p_CORR, utl_per_step_c_CORR, utl_per_step_r_CORR = get_utility_per_step(user, utl_per_step_p_CORR, utl_per_step_c_CORR, utl_per_step_r_CORR)
#             queries_answered_CORR = np.vstack((queries_answered_CORR, np.zeros(3)))
#             curr_idx = queries_answered_CORR.shape[0]
#             queries_answered_CORR[curr_idx - 1, 0] = user['dataset pairwise'].datapoints.shape[0]
#             queries_answered_CORR[curr_idx - 1, 1] = user['dataset clustering'].datapoints.shape[0]
#             queries_answered_CORR[curr_idx - 1, 2] = user['dataset ranking'].datapoints.shape[0]
#
#         # (6) understanding
#         user_underst = user['survey results']['understanding']
#         understanding[0, ['YES', 'NO', 'DUNNO'].index(user_underst[0])] += 1
#         understanding[1, ['YES', 'NO', 'DUNNO'].index(user_underst[1])] += 1
#         understanding[2, ['YES', 'NO', 'DUNNO'].index(user_underst[2])] += 1
#
#         # print any comments
#         comment = user['survey results']['comment']
#         # if len(comment) > 0:
#         #     print(user['name'])
#         #     print(comment)
#         #     print('')
#
#     # else:
#
#         # print(user['name'])
#         # print(user['tutorial pairwise'], user['tutorial clustering'], user['tutorial ranking'])
#         # print(user['experiment jobs pairwise'], user['experiment jobs clustering'], user['experiment jobs clustering'])
#
# print('')
# print(len(successful_users), 'people participated')
# print(utl_per_step_p_CORR.shape[0], 'understood the task ({}%)'.format(int(utl_per_step_p_CORR.shape[0]/len(successful_users)*100)), '\n')
# # print("queries answered:", queries_answered)
#
# # - - - - effort - - - -
#
# plt.figure(figsize=(6, 1.6))
#
# plt.barh([2, 1, 0], effort[:, 0] + effort[:, 1], align='center', color='tomato', ecolor='black', label='high')
# plt.barh([2, 1, 0], effort[:, 0], align='center', color='mediumseagreen', ecolor='black', label='okay')
#
# plt.yticks([2, 1, 0], ['pairwise', 'clustering', 'ranking'], fontsize=15)
#
# plt.xticks([0, len(successful_users)/4, len(successful_users)/2, len(successful_users)/4*3, len(successful_users)], ['0%', '25%', '50%', '75%', '100%'], size=15)
# plt.xlim([0, len(successful_users)])
# # plt.xlabel('%', fontsize=15)
#
# # putting the legend outside the box
# plt.legend(bbox_to_anchor=(1.01, 1.1), loc='upper left', fontsize=15)
# plt.gcf().tight_layout(rect=(0, 0, 0.8, 1))
#
# plt.savefig('eval_plots/effort')
# plt.close()
#
# # - - - - preferences - - - -
#
# plt.figure(figsize=(6, 1.6))
#
# plt.barh([2, 1, 0], preferences[:, 2]+preferences[:, 1]+preferences[:, 0], color='slateblue', label='3rd')
# plt.barh([2, 1, 0], preferences[:, 1]+preferences[:, 0], color='orchid', label='2nd')
# plt.barh([2, 1, 0], preferences[:, 0], color='gold', label='1st')
#
# plt.yticks([2, 1, 0], ['pairwise', 'clustering', 'ranking'], size=15)
#
# plt.xticks([0, len(successful_users)/4, len(successful_users)/2, len(successful_users)/4*3, len(successful_users)], ['0%', '25%', '50%', '75%', '100%'], size=15)
# plt.xlim([0, len(successful_users)])
# # plt.xlabel('%', fontsize=15)
#
# # putting the legend outside the box
# plt.legend(bbox_to_anchor=(1.01, 1.1), loc='upper left', fontsize=15)
# plt.gcf().tight_layout(rect=(0, 0, 0.8, 1))
#
# plt.savefig('eval_plots/preference')
# plt.close()
#
# # - - - - understanding - - - -
#
# plt.figure(figsize=(6, 1.6))
#
# plt.barh([2, 1, 0], understanding[:, 2]+understanding[:, 0]+understanding[:, 1], label='?', color='slategray')
# plt.barh([2, 1, 0], understanding[:, 1]+understanding[:, 0], label='NO', color='crimson')
# plt.barh([2, 1, 0], understanding[:, 0], label='YES', color='forestgreen')
#
# plt.yticks([2, 1, 0], ['pairwise', 'clustering', 'ranking'], size=15)
#
# plt.xticks([0, len(successful_users)/4, len(successful_users)/2, len(successful_users)/4*3, len(successful_users)], ['0%', '25%', '50%', '75%', '100%'], size=15)
# plt.xlim([0, len(successful_users)])
# # plt.xlabel('%', fontsize=15)
#
# # putting the legend outside the box
# plt.legend(bbox_to_anchor=(1.01, 1.1), loc='upper left', fontsize=15)
# plt.gcf().tight_layout(rect=(0, 0, 0.8, 1))
#
# plt.savefig('eval_plots/user_felt_understood')
# plt.close()
#
#
# # - - - - utility per timestep - - - -
#
# plt.figure(figsize=(6, 3))
#
# plt.plot(range(1, utl_per_step_p.shape[1]+1), np.average(utl_per_step_p, axis=0), color='cornflowerblue', label='pairwise', linewidth=2)
# plt.plot(np.zeros(100) + np.average(queries_answered, axis=0)[0], np.linspace(-0.1, 1.1, 100), '--', color='cornflowerblue')
#
# plt.plot(range(1, utl_per_step_c.shape[1]+1), np.average(utl_per_step_c, axis=0), color='firebrick', label='clustering', linewidth=2)
# plt.plot(np.zeros(100) + np.average(queries_answered, axis=0)[1], np.linspace(-0.1, 1.1, 100), '--', color='firebrick')
#
# plt.plot(range(1, utl_per_step_r.shape[1]+1), np.average(utl_per_step_r, axis=0), color='forestgreen', label='ranking', linewidth=2)
# plt.plot(np.zeros(100) + np.average(queries_answered, axis=0)[2], np.linspace(-0.1, 1.1, 100), '--', color='forestgreen')
#
# plt.plot(-1, -1, 'k--', label='avg')
#
# plt.xlim([0.5, np.max([utl_per_step_p.shape[1], utl_per_step_c.shape[1], utl_per_step_r.shape[1]]) + 0.5])
# plt.ylim([0.1, 0.75])
# plt.xlabel('query number', fontsize=15)
# plt.ylabel('utility', fontsize=15)
# plt.legend(fontsize=15)
# plt.tight_layout()
#
# plt.savefig('eval_plots/utility_per_timestep')
# plt.close()
#
# # - - - - utility per timestep (CORRECT) - - - -
#
# plt.figure()
# plt.title('true utility per timestep (for users that understood the task)')
#
# plt.plot(range(1, utl_per_step_p_CORR.shape[1] + 1), np.average(utl_per_step_p_CORR, axis=0), 'g', label='pairwise')
# plt.plot(np.zeros(100) + np.average(queries_answered_CORR, axis=0)[0], np.linspace(-0.1, 1.1, 100), 'g--', label='avg')
#
# plt.plot(range(1, utl_per_step_c_CORR.shape[1] + 1), np.average(utl_per_step_c_CORR, axis=0), 'b', label='clustering')
# plt.plot(np.zeros(100) + np.average(queries_answered_CORR, axis=0)[1], np.linspace(-0.1, 1.1, 100), 'b--', label='avg')
#
# plt.plot(range(1, utl_per_step_r_CORR.shape[1] + 1), np.average(utl_per_step_r_CORR, axis=0), color='orange', label='ranking')
# plt.plot(np.zeros(100) + np.average(queries_answered_CORR, axis=0)[2], np.linspace(-0.1, 1.1, 100), '--', color='orange', label='avg')
#
# plt.xlim([0.5, np.max([utl_per_step_p_CORR.shape[1], utl_per_step_c_CORR.shape[1], utl_per_step_r_CORR.shape[1]]) + 0.5])
# plt.ylim([0, 1])
# plt.xlabel('query number')
# plt.ylabel('utility')
# plt.legend()
# plt.tight_layout()
#
# plt.savefig('eval_plots/utility_per_timestep_CORRECT')
# plt.close()
