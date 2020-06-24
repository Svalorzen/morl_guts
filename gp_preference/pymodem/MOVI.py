#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:54:41 2016

@author: Diederik M. Roijers (University of Oxford)

This module contains the multi-objective value iteration algorithms:
the abstract multi-objective value iteration algorithm (MOVI)
which is parameterised by two pruning operators and a stop
criterion function based on the comparison of two value vector
sets and a threshold. When initialised with the proper pruning
operators (e.g. pruners.c_prune and pruners.pareto_prune) this leads
to the following well-known algorithms:
    - convex hull value iteration (CHVI) [Barrett & Narayanan (2008)]
    - Pareto value iteration (PVI) [White (1982)]
"""
import numpy
import Value


def multiobjective_value_iteration(momdp, prune1, prune2,
                                   differenceFunction, threshold, maxiter):
    """
    This function does multi-objective variable elimination.
    For a description of the algorithm + examples see our tutorial slides:
        Multi-Objective Decision Making
        Shimon Whiteson & Diederik M. Roijers
        European Agent Systems Summer School (EASSS)
        Catatia (Italy), 25th July 2016
        http://roijers.info/pub/easss.pdf
    Part 2: from slide 62 (pdf numbering 91) onwards (CHVI)
    """
    #####################################
    # first initialise the value function#
    #####################################
    lst = []
    nObj = momdp.n_objectives
    for i in range(nObj):
        lst.append(0.0)
    vec = numpy.array(lst)
    V = Value.ValueFunction(momdp.n_states)
    V.addVectorToAllSets(vec)
    ############################################################
    # then do Bellman backups until the value function converges#
    ############################################################
    cnt = 0
    difference = True
    while (cnt < maxiter) and difference:
        V2 = do_bellman_backup_all_states(momdp, prune1, prune2, V)
        diff = differenceFunction(V, V2)
        cnt += 1
        difference = diff > threshold
        V = V2
    return V


def do_bellman_backup(state, momdp, prune1, prune2, valueFunction):
    """
    This function performs a multi-objective Bellman backup for a
    given state, and a given (previous) value function.
    Note that this value function consists of sets of value vectors
    per state, and is an intermediate coverage set
    """
    ##############################################################
    # V' = prune1 (Union_a R(s,a) +                              #
    #    (Fold(prune2 after crosssum, s') gamma*T(s,a,s') V(s')))#
    ##############################################################
    nActions = momdp.n_actions
    gamma = momdp.discountFactor()
    Vs = Value.ValueVectorSet()
    for i in range(nActions):
        tList = momdp.getTransitionProbabilities(state, i)
        rExpect = momdp.expected_reward(state, i)
        # DEBUG: print("R(s,"+str(i)+"): " + str(rExpect))
        # DEBUG: print(str(tList))
        # DEBUG: print("\n")
        Vnew = Value.ValueVectorSet()
        for j in range(len(tList)):
            factor = gamma*tList[j]
            Vsprime = valueFunction.getValue(j)
            Vtrans = Vsprime.multiplyByScalar(factor)
            Vnew = Vnew.crossSum(Vtrans)
            Vnew = prune2(Vnew)
        Vnew = Vnew.translate(rExpect)
        Vs.addAll(Vnew.set)
    Vs = prune1(Vs)
    return Vs


def do_bellman_backup_all_states(momdp, prune1, prune2, valueFunction):
    Vnew = Value.ValueFunction(momdp.n_states)
    for i in range(momdp.n_states):
        Vi = do_bellman_backup(i, momdp, prune1, prune2, valueFunction)
        Vnew.setValue(i, Vi)
    return Vnew
