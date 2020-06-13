import numpy as np
import pandas as pd
from itertools import permutations
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd


# Assembly connection information
connections = {'c': {'skills': ['screwing', 'inserting'], 'tools': ['screwdriver'],
                     'parts': ['long_column', 'short_column']},
               's': {'skills': ['screwing'], 'tools': ['allen_key'], 'parts': ['shelf']}}


# New custom distance metric (min shifts, max(total shift cost) = 3, add=1, remove=1, shift=1)
def transform_dist(seq1, seq2):
    cost = 0
    for element in set(seq1):
        indices1 = [i for i, e in enumerate(seq1) if e == element]
        indices2 = [i for i, e in enumerate(seq2) if e == element]
        cost = cost + abs(len(indices1) - len(indices2))

        shift_costs = []
        idx_max = max([indices1, indices2], key=len)
        idx_min = min([indices2, indices1], key=len)
        for indices in permutations(idx_max):
            shift_cost = 0
            for idx in range(len(idx_min)):
                shift_cost = shift_cost + min(abs(idx_min[idx] - indices[idx]), 3)
            shift_costs.append(shift_cost)
        cost = cost + min(shift_costs)

    indices = [i for i, e in enumerate(seq2) if e not in seq1]
    cost = cost + len(indices)

    return cost


def fast_transform_dist(seq1, seq2):
    cost = 0
    for element in set(seq1):
        indices1 = [i for i, e in enumerate(seq1) if e == element]
        indices2 = [i for i, e in enumerate(seq2) if e == element]
        cost = cost + abs(len(indices1) - len(indices2))

        idx_max = max([indices1, indices2], key=len)
        idx_min = min([indices2, indices1], key=len)
        shift_costs = []
        invalid_indices = []
        for idx in idx_min:
            shift_cost = [min(abs(i - idx), 3) for i in idx_max]
            if any(sc_0 == 0 for sc_0 in shift_cost):
                invalid_indices.append(shift_cost.index(0))
            else:
                if all(sc_3 == 3 for sc_3 in shift_cost):
                    cost = cost + 3
                else:
                    shift_costs.append(shift_cost)

        if shift_costs:
            shift_costs = np.array(shift_costs)
            shift_costs = np.delete(shift_costs, invalid_indices, 1)
            n_rows, n_cols = np.shape(shift_costs)
            if len(shift_costs) == 1:
                cost = cost + min(shift_costs[0])
            else:
                shift_cost = []
                for shift_cost_matrix in [shift_costs, np.flip(shift_costs, 1)]:
                    for offset in range(n_cols - n_rows + 1):
                        diag = shift_cost_matrix.diagonal(offset)
                        shift_cost.append(sum(diag))
                cost = cost + min(shift_cost)

    indices = [i for i, e in enumerate(seq2) if e not in seq1]
    cost = cost + len(indices)

    return cost


def distance_matrix(sequences):
    distance_metric = fast_transform_dist

    # Create a distance matrix
    dist = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            dist[i, j] = distance_metric(sequences[i], sequences[j])
            dist[j, i] = distance_metric(sequences[j], sequences[i])
            # print [i, j], diff[i, j], diff[j, i]

    return dist


def procedural_grouping(user_plan):
    count = 1
    procedure = []
    prev_action = user_plan[0][0]
    prev_parts = connections[prev_action]['parts']
    prev_tools = connections[prev_action]['tools']
    for i in range(1, len(user_plan)):
        action = user_plan[i][0]
        parts = connections[action]['parts']
        tools = connections[action]['tools']
        if tools == prev_tools:
            count += 1
        else:
            procedure.append(prev_action + str(count))
            count = 1
        prev_action = action
        prev_parts = parts
        prev_tools = tools
    procedure.append(prev_action + str(count))

    return procedure


# Returns event number
def event_length(event):
    return int(''.join(filter(lambda i: i.isdigit(), event)))


# Calculate cluster dispersion score
def calinski_harabasz_score(dist_matrix, k, labels):
    n_data = len(dist_matrix)
    C = float('inf')
    for x in range(len(labels)):
        avg_dist = np.mean(dist_matrix[x][:])
        if avg_dist < C:
            C = avg_dist
            C_user = x

    W_k, B_k = 0, 0
    for q in range(1, k+1):
        q_users = [i for i, l in enumerate(labels) if l == q]
        n_q = len(q_users)
        C_q = float('inf')
        for x in q_users:
            avg_dist = np.mean([dist_matrix[x][u] for u in q_users])
            if avg_dist < C_q:
                C_q = avg_dist
                Cq_user = x

        w_k = 0
        for x in q_users:
            w_k = w_k + (dist_matrix[x][Cq_user])**2
        W_k = W_k + w_k

        B_k = B_k + (n_q*(dist_matrix[C_user][Cq_user])**2)

    ch_score = (B_k / W_k) * ((n_data-k) / (k-1))

    return ch_score


class Robot:
    def __init__(self):
        """
        Find dominant clusters of event sequences and clusters within events
        """
        # Load user data from csv
        data = pd.read_csv("game_data.csv", sep='\s*,\s*', engine='python')
        users = list(set(data['gameId']))

        # Populate user plans and time taken to complete
        plans = []
        for user in users:
            user_data = data.loc[data['gameId'] == user]
            time = list(user_data['time'])

            # Select best time from final rounds
            final_time = time[-3:]
            idx = 4 + final_time.index(min(final_time))
            plan = list(user_data.iloc[idx, 4:])
            plans.append(plan)

        # Separate primary and secondary plans
        self.primary_plans, self.secondary_plans = [], []
        for plan in plans:
            primary_plan = [action for action in plan if (action[-1] == 'r' or action[-1] == 'l')]
            self.primary_plans.append(primary_plan)

            secondary_plan = []
            previous_index = 0
            for primary_action in primary_plan:
                current_index = plan.index(primary_action)
                if current_index == previous_index:
                    secondary_actions = ['wait']
                else:
                    secondary_actions = plan[previous_index:current_index]
                previous_index = current_index + 1
                secondary_plan.append(str(secondary_actions))
            self.secondary_plans.append(secondary_plan)

        # Calculate the procedure by clustering actions that use the same parts
        self.procedures = []
        for plan in self.primary_plans:
            procedure = procedural_grouping(plan)
            self.procedures.append(procedure)

        # All connections
        self.all_actions = set(self.primary_plans[0])

        # Find clusters in procedures
        dist = distance_matrix(self.procedures)
        dist_array = ssd.squareform(dist)
        f_link = linkage(dist_array)
        dispersion_score = 0
        for cluster_threshold in range(int(max(dist_array))):
            procedure_clusters = list(fcluster(f_link, cluster_threshold, criterion='distance'))
            k = len(set(procedure_clusters))
            if k > 1:
                ch_score = calinski_harabasz_score(dist, k, procedure_clusters)
                if ch_score > dispersion_score:
                    dispersion_score = ch_score
                    optimal_cluster_threshold = cluster_threshold
        self.procedure_clusters = list(fcluster(f_link, optimal_cluster_threshold, criterion='distance'))
        self.dominant_clusters = [label for label in set(self.procedure_clusters) if
                                  self.procedure_clusters.count(label) >= 2]

        # List all distinct events
        all_events = []
        dominant_users = [i for i, label in enumerate(self.procedure_clusters) if label in self.dominant_clusters]
        for dc_user in dominant_users:
            all_events = all_events + self.procedures[dc_user]
        self.all_events = list(set(all_events))

        # Find clusters in events
        self.dom_event_clusters, self.secondary_event_data, self.primary_event_data = {}, {}, {}
        for dc in self.dominant_clusters:
            dominant_users = [i for i, label in enumerate(self.procedure_clusters) if label == dc]
            dc_events = []
            for dc_user in dominant_users:
                dc_events = dc_events + self.procedures[dc_user]
            dc_events = list(set(dc_events))

            # Populate action sequences corresponding to each event
            self.secondary_event_data[str(dc)], self.primary_event_data[str(dc)] = {}, {}
            for event in dc_events:
                secondary_instances, primary_instances = [], []
                for usr in dominant_users:
                    idx = 0
                    for e in self.procedures[usr]:
                        if e == event:
                            sec_instance = self.secondary_plans[usr][idx:idx + event_length(e)]
                            secondary_instances.append(sec_instance)
                            pri_instance = self.primary_plans[usr][idx:idx + event_length(e)]
                            primary_instances.append(pri_instance)
                            idx = idx + event_length(e)
                        else:
                            idx = idx + event_length(e)
                self.secondary_event_data[str(dc)][event] = secondary_instances
                self.primary_event_data[str(dc)][event] = primary_instances

            # Find clusters within events
            self.dom_event_clusters[str(dc)] = {}
            for event in dc_events:
                dist = distance_matrix(self.secondary_event_data[str(dc)][event])
                dist_array = ssd.squareform(dist)
                f_link = linkage(dist_array)
                dispersion_score = 0
                for cluster_threshold in range(int(max(dist_array))):
                    event_clusters = list(fcluster(f_link, cluster_threshold, criterion='distance'))
                    k = len(set(event_clusters))
                    if k > 1:
                        ch_score = calinski_harabasz_score(dist, k, event_clusters)
                        if ch_score > dispersion_score:
                            dispersion_score = ch_score
                            optimal_cluster_threshold = cluster_threshold
                event_clusters = fcluster(f_link, optimal_cluster_threshold, criterion='distance')
                self.dom_event_clusters[str(dc)][event] = event_clusters

    def predict(self, action_subseq):
        """
        Predict the next action given the subset of actions observed so far.
        :param action_subseq: List of actions.
        :return: List of next actions that the robot should execute.
                 'lr' - large column, 'sm' - small column, 'sf' - shelf, 'wa' - wait.
        """
        # Separate primary actions from the action sub-sequence
        primary_subseq = [action for action in action_subseq if (action[-1] == 'r' or action[-1] == 'l')]
        secondary_subseq = []
        previous_index = 0
        for primary_action in primary_subseq:
            current_index = action_subseq.index(primary_action)
            if current_index == previous_index:
                secondary_actions = ['wait']
            else:
                secondary_actions = action_subseq[previous_index:current_index]
            previous_index = current_index + 1
            secondary_subseq.append(str(secondary_actions))

        # Convert sequence of primary actions to sequence of events
        sub_procedure = procedural_grouping(primary_subseq)

        # Find candidate events for the last event
        curr_event = sub_procedure[-1]
        curr_event_len = event_length(curr_event)
        all_curr_events = [e for e in self.all_events if e[0] == curr_event[0]]
        all_curr_events = [e for e in all_curr_events if curr_event_len <= event_length(e)]

        # Possible sequences of evemts
        sub_procedures = []
        for e in all_curr_events:
            sub_procedures.append(sub_procedure[:-1] + [e])

        # Predict the dominant cluster
        final_prob = 0
        for sp in sub_procedures:
            prob_sum, probs = 0, {}
            for dc in self.dominant_clusters:
                procedure_users = [i for i, label in enumerate(self.procedure_clusters) if label == dc]

                # Count the number of times the sub-sequence of events is observed in the dominant cluster
                n_pref = 0
                for p_user in procedure_users:
                    if sp == self.procedures[p_user][:len(sub_procedure)]:
                        n_pref += 1

                # Bayes rule
                # P(Dominant cluster|Event sub-sequence) = P(Event sub-sequence|Dominant cluster) * P(Dominant cluster)
                prob = float(n_pref + 1) / float(len(procedure_users) + 1)
                prob = prob * (float(self.procedure_clusters.count(dc)) / float(len(self.procedure_clusters)))
                prob_sum = prob_sum + prob
                probs[str(dc)] = prob

            # Normalization
            for dc in probs:
                prob = probs[dc] / prob_sum

                # Select the match with maximum probability
                if prob >= final_prob:
                    final_prob = prob
                    final_pref = dc
                    final_proc = sp

        # The most common event sequence in the dominant cluster
        pred_users = [i for i, label in enumerate(self.procedure_clusters) if label == int(final_pref)]
        pred_procedure = self.procedures[np.random.choice(pred_users)]

        # Remaining primary actions that the human can execute
        remaining_actions = self.all_actions - set(primary_subseq)

        # Use robot prediction only if the confidence is above 75%
        confidence_threshold = 0.75
        if final_prob > confidence_threshold:
            print("Prediction probability:", final_prob)

            # If the last event in sub-sequence is still ongoing
            if event_length(curr_event) < event_length(final_proc[-1]):
                pred_event = final_proc[-1]
                possible_event_clusters = list(self.dom_event_clusters[final_pref][pred_event])

                prev_primary_action = primary_subseq[-1 * curr_event_len]
                event_index = primary_subseq.index(prev_primary_action)
                secondary_subsubseq = secondary_subseq[event_index:event_index + curr_event_len]

                max_prob, prob_sum, probs = 0, 0, {}
                for dc in set(possible_event_clusters):
                    event_users = [i for i, label in enumerate(possible_event_clusters) if label == dc]
                    n_pref = 0
                    for e_user in event_users:
                        if secondary_subsubseq == \
                                self.secondary_event_data[final_pref][pred_event][e_user][:curr_event_len]:
                            n_pref += 1

                    # Bayes rule
                    # P(Event cluster|Action sub-sequence) = P(Action sub-sequence|Event cluster) * P(Event cluster)
                    prob = float(n_pref + 1) / float(len(event_users) + 1)
                    prob = prob * (float(possible_event_clusters.count(dc)) / float(
                        len(possible_event_clusters)))
                    prob_sum = prob_sum + prob
                    probs[str(dc)] = prob

                # Normalization
                for dc in probs:
                    prob = probs[dc] / prob_sum

                    # Select the match with maximum probability
                    if prob >= max_prob:
                        max_prob = prob
                        event_pref = dc

                # The most common action sequence in the event cluster
                pred_users = [i for i, label in enumerate(possible_event_clusters) if
                              label == int(event_pref)]
                possible_actions = [a[curr_event_len] for i, a in
                                    enumerate(self.secondary_event_data[final_pref][pred_event]) if i in pred_users]
                pred_actions = stats.mode(possible_actions)[0][0]

            # If the last event in sub-sequence has finished
            else:
                # The most common action sequence in the next event
                pred_event = pred_procedure[len(sub_procedure)]
                possible_actions = [a[0] for a in self.secondary_event_data[final_pref][pred_event]]
                pred_actions = stats.mode(possible_actions)[0][0]

            # Just change format of actions (custom shitty code)
            pred_actions = pred_actions[1:-1].split(', ')
            prediction = [part[1:3] for part in pred_actions]
            # prediction = [a for a in pred_actions if a not in action_subseq]

            print("Predicted response:")
            return prediction
        else:
            return [None]

    def test(self, timestep, user=0):
        """
        Test the prediction of the algorithm for a particular user and timestep.
        :param timestep: Integer (index) between 0 to 11.
        :param user: Integer (index) between 0 to 22. The user for whom you want to make predictions.
        :return: List of actions. 'lr' - large column, 'sm' - small column, 'sf' - shelf, 'wa' - wait.
        """
        action_seq = self.primary_plans[user][:timestep]
        print("Actual response:", self.secondary_plans[user][timestep])

        return self.predict(action_seq)


# Test Code
# r = Robot()
# print r.predict(['lr_right', 'sm_right', 'cr'])#, 'lr_left', 'sm_left', 'cl', 'sf1', 's1l', 'sf2', 's2l', 'sf3', 's3l', 'sf4', 's4l', 'sf5', 's5l'])

# print "Done"
