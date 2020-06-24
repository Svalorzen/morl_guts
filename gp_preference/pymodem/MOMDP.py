#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:25 2016

@author: Diederik M. Roijers (University of Oxford)
"""

import numpy


class MOMDP:
    """
    The following abstract class definition describes an MOMDP.
    In order to add an MOMDP, please implement all these abstract
    functions.
    """
    problemType = "MOMDP"

    def __init__(self):
        pass

    def get_problem_type(self):
        """
        Function to quickly identify the type of problem, e.g., an "MOMDP"
        """
        return self.problemType

    def expected_reward_SAS(self, state, action, nextState):
        """
        For planning purposes, one should use the expected reward: R(s,a,s')
        The reward should be a numpy array doubles
        of length self.n_objectives
        """
        pass

    def expected_reward(self, state, action):
        """
        For planning purposes, one should use the expected reward.
        R(s,a) should be equal to \sum_s' T(s,a,s') * R(s,a,s')
        The reward should be a numpy array doubles
        of length self.n_objectives()
        """
        pass

    def getTransitionProbabilities(self, state, action):
        """
        For planning purposes, the transition probabilities are required.
        This function should return a list of length self.n_states.
        Each element i of this list should contain the probability
        of transitioning to state i from state 'state' and action 'action',
        i.e., T(state, action, i)
        The list should contain double precision floats.
        """
        pass

    def getTransitionProbability(self, state, action, nextState):
        """
        For planning purposes, the transition probabilities are required.
        This function should return T(state, action, nextState).
        The return type is a double precision float.
        """
        pass

    def performAction(self, action):
        """
        For simulation and learning purposes, this function changes the
        state of the MOMDP, and returns a reward (double precision float).
        (NB: for deterministic reward functions the given reward should
        be equal to self.expected_reward(state, action, nextState), however
        for non-deterministic reward functions, this need not be so.)
        """
        pass


    def getCurrentState(self):
        """
        For simulation and learning purposes, this function returns the
        current state of the MOMDP.
        """
        pass

    def isInTerminalState(self):
        """
        For simulation and learning purposes, this function returns the
        whether the MOMDP is in a terminal state.
        """
        pass

    def reset(self):
        """
        For simulation and learning purposes, this function resets the
        current state of the MOMDP by sampling it from the initial state
        distribution (mu_0).
        """
        pass


#########################################################################
#########################################################################
################    NEW CLASS DEFINITION: TestMOMDP    ##################
#########################################################################
#########################################################################

class TestMOMDP(MOMDP) :
    """
    An MOMDP with 11 states:
        GL XX XX XX XX ST XX XX XX XX GR
    where:
        GL=goal left (0),
        ST=start (5),
        GR=goal right (10),
        XX=other states (1-4, 6-9)
    actions: left (0) and right (1), and stop (2)
    deterministic transitions:
        left  -> state--
        right -> state++
    deterministic rewards:
        R(_,_,GL) = [10.0, 0.0]
        R(_,_,GR) = [0.0, 20.0]
        R(not 0 or 10, stop,_) = [1.0, 1.0]
        and R(_,_,_) = 0 otherwise

    The point of this test MOMDP is that with Pareto planning/learning
    the value vector set for each state should at convergence contain
    3 vectors corresponding to the policies:
        - move all the way to the left
        - move all the way to the right
        - stop immediately
    However, with convex planning/learning, there should be only 2
    value vectors per state, as "stop immediately" leads to a (1,1)
    value vector, which is C-dominated by the other two policies.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_states = 11
        self.n_objectives = 2
        self.n_actions = 3
        self.discount_factor = 0.9
        self.currentState = 5

    def performAction(self, action):
        if self.currentState == 0 or self.currentState == 10:
            return numpy.array([0.0, 0.0])
            # this is a terminal state
        if action == 0:
            self.currentState -= 1
            if self.currentState == 0:
                # self.reset()
                return numpy.array([10.0, 0.0])
        if action == 1:
            self.currentState += 1
            if self.currentState == 10:
                # self.reset()
                return numpy.array([0.0, 20.0])
        if(action == 2):
            self.currentState = 0
            return numpy.array([1.0, 1.0])
        return numpy.array([0.0, 0.0])

    def getCurrentState(self):
        return self.currentState

    def getTransitionProbability(self, state, action, nextState):
        if state == 0 or state == 10:
            # this is a terminal state
            if state == nextState:
                return 1.0
            else:
                return 0.0
        if action == 0:
            if nextState == state-1:
                return 1.0
        if action == 1:
            if nextState == state+1:
                return 1.0
        if action == 2:
            if nextState == 0:
                return 1.0
        return 0.0

    def expected_reward(self, state, action):
        if action == 0:
            if state == 1:
                return numpy.array([10.0, 0.0])
        if action == 1:
            if state == 9:
                return numpy.array([0.0, 20.0])
        if action == 2 and not (state == 0 or state == 10):
            return numpy.array([1.0, 1.0])
        return numpy.array([0.0, 0.0])  # including terminal states

    def expected_reward_SAS(self, state, action, nextState):
        return self.expected_reward(state, action)

    def getTransitionProbabilities(self, state, action):
        result = []
        for i in range(11):
            dbl = self.getTransitionProbability(state, action, i)
            result.append(dbl)
        return result

    def isInTerminalState(self):
        if self.currentState == 0 or self.currentState == 10:
            return True
        else:
            return False
