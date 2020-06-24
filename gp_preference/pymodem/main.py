#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:41:56 2016

@author: Diederik M. Roijers (University of Oxford)
"""

from .MOMDP import TestMOMDP
import numpy as np
from . import Value, pruners, PointMOVI

def printme( str ):
   #This prints a passed string into this function
   print(str)


# Create a test MOMDP, to check that everything is working
m = TestMOMDP()

"""
# This codes runs roll-outs with a given MOMDP
# Useful to test new MOMDPs

s = m.get_problem_type()
print("\n",s)
print("start state: ", m.currentState)
for i in range(10):
    print("\n")
    rndAction = 1
    print("current state: ", m.currentState, "\n")
    print("action: ", rndAction, "\n")
    print("T(s,a,_) = ", m.getTransitionProbabilities(
          m.currentState,rndAction), "\n")
    reward = m.performAction(rndAction)
    print("reward: ", reward, "\n")
    print("next state: ", m.currentState, "\n")
    if(m.isInTerminalState()):
        m.reset()
print("\n done")

print("\n ", m.number_of_objectives())
"""

#Test program: perform CHVI
#V = MOVI.multiobjective_value_iteration(m, pruners.c_prune,
#                                      pruners.c_prune,
#                                      Value.maxDifferenceAcrossValueFunctions
#                                      , 0.01, 30)
#print(V.__str__())

vvs1 = Value.testValueVectorSet()
print(vvs1.__str__())
#PointMOVI.globalPoints = [[0.6,0.4],[0.5,0.5],[0.2,0.8]]
vvs2, vals = PointMOVI.globalPointBasedPrune(vvs1)
print(vvs2.__str__())
print(vals)


vvs1 = Value.ValueVectorSet()
vvs1.add(np.array([1.0,0]))
vvs1.add(np.array([0.0,1.0]))
vvs1.add(np.array([0.75,0.75]))
vvs2 = pruners.max_prune(vvs1, [0.3,0.7])
print(vvs2.__str__())



"""
# The code below is just testing some operations on sets of value vectors
# and executing some sample linear programs.

c = [0, 0, -1]
A = [[0.75, -0.25, 1], [-0.25, 0.75, 1]]
b = [0, 0]
Aeq = [[1,1,0]]
beq = [1]
w0_bnds = (0, 1)
w1_bnds = (0, 1)
x_bnds = (None, None)
res = linprog(c, A, b, Aeq, beq, bounds=[w0_bnds, w1_bnds, x_bnds])
print(res.fun)


vvs1 = Value.ValueVectorSet()
vvs1.add(np.array([1.0,0]))
vvs1.add(np.array([0.0,1.0]))
vvs1.add(np.array([0.75,0.75]))
tup = vvs1.removeMaximisingLinearScalarisedValue([1.0,0.0])
vvs1 = tup[1]
print("maximising: "+ str(tup[0]))
print(vvs1.__str__())


vvs1 = Value.ValueVectorSet()
vvs1.add(np.array([1.0,0]))
vvs1.add(np.array([0.0,1.0]))
vvs1.add(np.array([0.5,0.5]))
vvs2 = pruners.c_prune(vvs1)
print(vvs1.__str__())
print(vvs2.__str__())

loc = np.array([1.0,2.0])

vel = np.array([0.0,2.1])
print(pruners.weak_pareto_dominates(loc,vel))


vvs1 = Value.ValueVectorSet()
vvs2 = Value.ValueVectorSet()
vvs1.add(np.array([1.0,1.0]))
vvs1 = vvs1.multiplyByScalar(0.5)
vvs1.add(np.array([2.0,0.0]))
print(vvs1.__str__())
vvs3 = vvs1.translate(np.array([0.5,0.5]))
print(vvs3.__str__())
print(Value.maxDifferenceAcrossObjectivesSet(vvs1, vvs3))
vvs2.add(np.array([1.0,0.5]))
vvs2.add(np.array([0.0,2.0]))
vvs3 = vvs1.crossSum(vvs2)
printme(vvs3.__str__())
printme(pruners.pareto_prune(vvs3).__str__())
"""
