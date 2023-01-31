#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script that prints the Pareto front of a given file
"""

import numpy as np
import matplotlib.pyplot as pl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f1", dest="Filename1", type=str,
                    help="Filename with front calculated with SMS-EGO")
parser.add_argument("-f2", dest="Filename2", type=str,
                    help="Filename with front calculated with NSGAII")
parser.add_argument("--tpf", dest="TPF", type=str,
                    help="true Pareto front", default=None,
                    required=False)

args = parser.parse_args()

if args.TPF is not None:
    F1, F2 = np.loadtxt(args.TPF, unpack=True)
    ISorted = np.argsort(F1)
    f1 = F1[ISorted]
    f2 = F2[ISorted]
else:
    f1 = np.linspace(0, 1, 1000)
    f2 = 1 - np.sqrt(f1)

if args.Filename1[-3:] == "npz":
    Data = np.load(args.Filename1)
    F1D1, F2D1 = Data["Front"][:, 0], Data["Front"][:, 1]
    I2D1 = np.argsort(F1D1)

else:
    raise TypeError("Extension should be npz")

if args.Filename2[-3:] == "npz":
    Data = np.load(args.Filename2)
    F1D2, F2D2 = Data["Front"][:, 0], Data["Front"][:, 1]
    I2D2 = np.argsort(F1D2)

else:
    raise TypeError("Extension should be npz")

pl.plot(F1D1[I2D1], F2D1[I2D1], 'o--', label="SMS-EGO")
pl.plot(F1D2[I2D2], F2D2[I2D2], 'o--', label="NSGAII")
#if args.TPF is not None:
pl.plot(f1, f2, 'o', label="TPF")
pl.legend()

pl.xlabel(r"$f_1$")
pl.ylabel(r"$f_2$")

pl.grid()

pl.show()
