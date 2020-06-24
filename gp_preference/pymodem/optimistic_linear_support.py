from priority_queue_set import PriorityQueueSet
from MOMDP import TestMOMDP
from pruners import max_prune, c_prune
from Value import ValueVectorSet, maxDifferenceAcrossObjectivesSet
from Value import inner_product
import numpy as np
from collections import defaultdict
import itertools

import pulp


class OptimisticLinearSupport():
    """
    """
    def __init__(self, momdp, scalarized_solver):
        """
        """
        self.momdp = momdp
        self.scalarized_solver = scalarized_solver
        self.queue = PriorityQueueSet()
        self.solution_set = ValueVectorSet()
        self.corner_to_value = defaultdict(list)
        self.value_to_corner = defaultdict(list)

    def initialize_ols(self):
        """
        Initializes Optimistic Linear Support
        """
        self.n_objectives = 3  # self.momdp.n_objectives
        print("MOMDP has {} objectives.".format(self.n_objectives))
        for i in range(self.n_objectives):
            weight_vector = [0.0 for x in range(self.n_objectives)]
            weight_vector[i] = 1.0
            # Initialize current value vector to have inf value on objectives
            value_vector = np.array([-float("inf") for x in
                                     range(self.n_objectives)])
            priority = float("inf")
            print("Adding weight vector with {} to the queue. \n "
                  "{}".format(priority, weight_vector))
            self.queue.add((priority, tuple(weight_vector)))
            self.corner_to_value[tuple(weight_vector)].append(value_vector)
            # self.value_to_corner[tuple(value_vector)] = weight_vector

    def run_ols(self):
        """
        Runs Optimistic Linear Support
        """
        threshold = 0.01
        while not self.queue.empty():
            print("State of the queue:\n {}".format(self.queue.peak_all()))
            print("State of w2v: \n{}".format(self.corner_to_value))
            print("State of sS : \n{}".format(self.solution_set))
            item = self.queue.get()

            print("WV: {}".format(item))
            weight_vector = item[1]
            print("WVDICT: {}".format(self.corner_to_value[weight_vector]))
            value_vector_set = self.scalarized_solver(self.momdp,
                                                      weight_vector)
            value_vector = value_vector_set.set[-1]
            prod = inner_product(value_vector, weight_vector)
            other_prod = inner_product(self.corner_to_value[weight_vector][-1],
                                       weight_vector)

            print("VV: {}".format(value_vector))
            if self.solution_set.set:
                # diff = maxDifferenceAcrossObjectivesSet(value_vector_set,
                #                                        self.solution_set)
                diff = prod - other_prod
                if diff > threshold:
                    print("Current: {}".format(weight_vector))
                    self.remove_from_value_dict(weight_vector)
                    obsolete_weights = self.check_obsolete(value_vector,
                                                           weight_vector)
                    # new corner
                    new_weights = self.new_corner_weights(value_vector_set,
                                                          obsolete_weights,
                                                          self.solution_set)
                    self.solution_set.addAll(value_vector_set.set)
            else:
                # No new corner weights available when this is the first
                # new vector.
                peaked = self.queue.peak_all()
                for item in peaked:
                    wv = item[1]
                    self.corner_to_value[tuple(wv)].append(value_vector)
                    self.value_to_corner[tuple(value_vector)].append(wv)
                self.solution_set.addAll(value_vector_set.set)
        return self.solution_set

    def new_corner_weights(self, value_vector_new, obsolete_weights,
                           solution_set):
        """
        Retrieve new corner weights
        """
        boundaries = self.get_boundaries(obsolete_weights)
        print(boundaries)
        new_weights = []
        old_value_vectors = []
        for weight in obsolete_weights:
            found_values = self.corner_to_value[weight]
            for v in found_values:
                v = tuple(v)
                if v not in old_value_vectors and not v == value_vector_new:
                    old_value_vectors.append(v)
        for i in boundaries:
            old_value_vectors.append(i)
        permutation_list = list(itertools.combinations(old_value_vectors,
                                                       self.n_objectives-1))
        print("plist: {}".format(permutation_list))
        for permutation in permutation_list:
            matrix, boundaries = self.permutation_matrix(permutation,
                                                         value_vector_new.set)
            s, found_weights, value = self.compute_weights(matrix,
                                                           obsolete_weights,
                                                           boundaries)
            if not s:
                continue
            new_vector = value_vector_new.set[-1]
            self.value_to_corner[tuple(new_vector)].append(found_weights)
            self.corner_to_value[tuple(found_weights)].append(new_vector)
            priority = 1.0
            print("Adding: {}".format(found_weights))
            self.queue.add((priority, tuple(found_weights)))
        print(self.queue.peak_all())
        for w in obsolete_weights:
            if not self.extremum_check(w):
                self.queue.remove_item(w)
        print(self.queue.peak_all())

        print("".join("-" for i in range(50)))

        for vector in self.value_to_corner:
            print("VECTOR: ", vector, "\nWEIGHTS: ",
                  self.value_to_corner[vector])

        print("".join("-" for i in range(50)))
        for weight in self.corner_to_value:
            print("WEIGHT: ", weight, "\nVECTORS: ",
                  self.corner_to_value[weight])

        print("".join("-" for i in range(50)))

        return None

    def compute_weights(self, matrix, weights, boundaries):
        """
        """
        solutions = np.zeros((matrix.shape[0],))
        solutions[-1] = 1
        print('matrix:\n {}'.format(matrix))
        print('soluti:\n {}'.format(solutions))
        w_sol = np.linalg.solve(matrix, solutions)
        for i in boundaries:
            w_sol = np.insert(w_sol, i, 0)
        within_simplex = self.simplex_check(w_sol[:-1])
        extremum = self.extremum_check(w_sol[:-1])
        has_nans = np.isnan(w_sol)
        print("Wsol: {}".format(w_sol))
        print("Within simplex: {}, extremum: {}".format(within_simplex,
                                                        extremum))
        print("Has nans: {}".format(has_nans))
        s = within_simplex and not extremum
        s = s and True not in has_nans
        return s, w_sol[:-1], w_sol[-1]

    def simplex_check(self, vector):
        """
        """
        for i in vector:
            if i < 0:
                return False
        return sum(vector) == 1.0

    def permutation_matrix(self, permutation, value_vector):
        """
        """
        value_vector_chosen = np.append(value_vector[-1], -1)
        matrix_list = [value_vector_chosen]
        boundaries = []
        for element in permutation:
            if type(element) == int:
                # now we found a boundary
                boundaries.append(element)
            else:
                element = np.append(element, -1)
                matrix_list.append(element)
        simplex_constraint = np.ones((self.n_objectives+1,))
        simplex_constraint[-1] = 0
        matrix_list.append(simplex_constraint)
        if len(matrix_list) > 1:
            matrix_list = np.vstack(matrix_list)
            matrix_list = np.delete(matrix_list, boundaries, 1)

        return matrix_list, boundaries

    def check_obsolete(self, new_value_vector, checked_weights):
        """
        Check for obsolete corner weights
        """
        original_value = inner_product(new_value_vector, checked_weights)
        weight_list = [checked_weights]
        obsolete_list = []
        while weight_list:
            weight = tuple(weight_list[-1])
            del weight_list[-1]
            value_vector_set = self.corner_to_value[tuple(weight)]
            if not value_vector_set:
                continue
            value_vector_0 = value_vector_set[-1]
            scalarized_value = inner_product(value_vector_0, weight)
            if original_value > scalarized_value:
                for value_vector in value_vector_set:
                    current_weight = self.value_to_corner[tuple(value_vector)]
                    for w in current_weight:
                        w = tuple(w)
                        if not (w == weight) and not (w in obsolete_list):
                            weight_list.append(w)
                obsolete_list.append(weight)
        print("Obsolete: {}".format(obsolete_list))
        for vectors in self.value_to_corner:
            weights = self.value_to_corner[vectors]
            for w in weights:
                w = tuple(w)
                if w in obsolete_list and not self.extremum_check(w):
                    temp_list = [tuple(x) for x in
                                 self.value_to_corner[vectors]]
                    element_index = temp_list.index(w)
                    self.value_to_corner[vectors].pop(element_index)
        return obsolete_list

    def remove_from_value_dict(self, weight_vector):
        """
        """
        weight_vector = tuple(weight_vector)
        vectors = self.corner_to_value[weight_vector]
        for v in vectors:
            v = tuple(v)
            temp_list = [tuple(x) for x in
                         self.value_to_corner[v]]
            if weight_vector in temp_list and not self.extremum_check(weight_vector):
                index = temp_list.index(weight_vector)
                self.value_to_corner[v].pop(index)

    def get_boundaries(self, weights):
        """
        Return the indices of objectives for which one of the weights has a 0
        """
        indices = []
        for obj in range(self.n_objectives):
            for w in weights:
                if w[obj] == 0:
                    indices.append(obj)
                    break
        return indices

    def extremum_check(self, w):
        """
        """
        if 1.0 in w:
            return True
        else:
            return False

    def proper_diff_check(self, vvs1, vvs2):
        """
        """



if __name__ == "__main__":
    vvs1 = ValueVectorSet()
    vvs1.add(np.array([1.0, 0, 0]))
    vvs1.add(np.array([0.0, 1.0, 0]))
    vvs1.add(np.array([0.75, 0.75, 0]))
    vvs1.add(np.array([0.5, 0.5, 0]))
    vvs1.add(np.array([0.25, 0, 0.76]))
    vvs1.add(np.array([0.15, 0.9, 0.76]))
    vvs1.add(np.array([0.24, 0, 0.76]))
    vvs1.add(np.array([0.25, 0.04, 0.76]))
    vvs1.add(np.array([0.9, 0.01, 0.16]))
    vvs1.add(np.array([0.2, 0.4, 0.4]))
    vvs1.add(np.array([0.05, 0.63, 0.46]))
    t_momdp = vvs1
    scalarized_solver = max_prune
    ols = OptimisticLinearSupport(t_momdp, scalarized_solver)
    ols.initialize_ols()
    ols_solutions = ols.run_ols()
    print("OLS SOLUTIONS:\n {}".format(ols_solutions))
    cpr_solutions = c_prune(vvs1)
    print("CPR SOLUTIONS:\n {}".format(cpr_solutions))

    #pulp.pulpTestAll()


    # declare your variables
    x1 = pulp.LpVariable("x1", 0, 40)   # 0<= x1 <= 40
    x2 = pulp.LpVariable("x2", 0, 1000) # 0<= x2 <= 1000

    # defines the problem
    prob = pulp.LpProblem("problem", pulp.LpMaximize)

    # defines the constraints
    prob += 2*x1+x2 <= 100
    prob += x1+x2 <= 80
    prob += x1<=40
    prob += x1>=0
    prob += x2>=0

    # defines the objective function to maximize
    prob += 3*x1+2*x2

    # solve the problem
    status = prob.solve()
    pulp.LpStatus[status]

    # print the results x1 = 20, x2 = 60
    print(x1.value())
    print(x2.value())
