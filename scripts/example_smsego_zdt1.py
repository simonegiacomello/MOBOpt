#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#example_zdt1.py
import numpy as np
import matplotlib.pyplot as pl

import mobopt as mo
import deap.benchmarks as db
import argparse


def target(x):
    return np.asarray(db.zdt1(x))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="ND", type=int, metavar="ND",
                        help="Number of Dimensions for ZDT1",
                        default=30,
                        required=False)
    parser.add_argument("-i", dest="NI", type=int, metavar="NI",
                        help="Number of iterations of the method",
                        required=True)
    parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                        help="Number of initialization points",
                        required=False, default=5)
    parser.add_argument("-v", dest="verbose", action='store_true',
                        help="Verbose")
    parser.add_argument("--filename", dest="Filename", type=str,
                        default="ZDT1.dat",
                        help="Filename for saving data")
    parser.set_defaults(Reduce=False)

    args = parser.parse_args()

    NParam = args.ND
    NIter = args.NI
    N_init = args.NInit
    verbose = args.verbose

    PB = np.asarray([[0, 1]]*NParam)

    f1 = np.linspace(0, 1, 1000)
    f2 = (1-np.sqrt(f1))

    Optimize = mo.MOBayesianOpt(target=target,
                                NObj=2,
                                pbounds=PB,
                                Picture=True,
                                MetricsPS=False,
                                TPF=np.asarray([f1, f2]).T,
                                verbose=verbose,
                                Filename=args.Filename,
                                max_or_min='min')

    Optimize.initialize(init_points=N_init)

    front, pop = Optimize.maximize_smsego(n_iter=NIter)
    PF = np.asarray([np.asarray(y) for y in Optimize.y_Pareto])
    PS = np.asarray([np.asarray(x) for x in Optimize.x_Pareto])

    FileName = "SMS-EGO_" + args.Filename
    np.savez(FileName,
             Front=-front,
             Pop=pop,
             PF=PF,
             PS=PS)


    fig, ax = pl.subplots(1, 1)
    ax.plot(f1, f2, '-', label="TPF")
    ax.scatter(-front[:, 0], -front[:, 1], label=r"$\chi$")
    ax.grid()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.legend()
    fig.savefig(FileName+".png", dpi=300)

    GenDist = mo.metrics.GD(front, np.asarray([f1, f2]).T)
    Delta = mo.metrics.Spread2D(front, np.asarray([f1, f2]).T)

    if verbose:
        print("GenDist = ", GenDist)
        print("Delta = ", Delta)

    pass


if __name__ == '__main__':
    main()