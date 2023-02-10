#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script that prints the Pareto front of a given file
"""

import numpy as np
import matplotlib.pyplot as pl
import targets
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target", dest="target", type=str,
                    help="target function", default=None,
                    required=False)

args = parser.parse_args()
target = targets.target(args.target.lower())
f1 = target.f1
f2 = target.f2

Filename1 = "SMS-EGO_" + args.target + ".dat.npz"
Filename2 = "NSGAII_" + args.target + ".dat.npz"

Data = np.load(Filename1)
F1D1, F2D1 = Data["Front"][:, 0], Data["Front"][:, 1]
I2D1 = np.argsort(F1D1)

Data = np.load(Filename2)
F1D2, F2D2 = Data["Front"][:, 0], Data["Front"][:, 1]
I2D2 = np.argsort(F1D2)


pl.plot(F1D1[I2D1], F2D1[I2D1], 'o--', label="SMS-EGO")
pl.plot(F1D2[I2D2], F2D2[I2D2], 'o--', label="NSGAII")

pl.plot(f1, f2, '-', label="TPF")
pl.legend()
pl.title(args.target)
pl.xlabel(r"$f_1$")
pl.ylabel(r"$f_2$")

pl.grid()

pl.show()