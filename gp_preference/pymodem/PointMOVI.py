#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:10:06 2016

@author: Diederik M. Roijers (Vrije Universiteit Brussel)
"""

import Value


globalPoints = []


def pointBasedPrune(vvSet, points=[]):
    """
    Returns a new ValueVectorSet from which all vectors that are not
    optimal for any weight in `points' are removed, as well as a list
    of the scalarised values at each eight in `points'
    """
    result = Value.ValueVectorSet()
    scalVals = []
    for i in range(len(points)):
        tup = vvSet.removeMaximisingLinearScalarisedValue(points[i])
        result.add(tup[0])
        scalval = Value.inner_product(tup[0], points[i])
        scalVals.append(scalval)
    return result, scalVals


def globalPointBasedPrune(vvSet):
    return pointBasedPrune(vvSet, globalPoints)
