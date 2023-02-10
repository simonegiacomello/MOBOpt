#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#example_smsego.py
import numpy as np
import matplotlib.pyplot as pl
import targets
import mobopt as mo
import argparse



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="ND", type=int, metavar="ND",
                        help="Number of Dimensions",
                        default=30,
                        required=False)
    parser.add_argument("-i", dest="NI", type=int, metavar="NI",
                        help="Number of iterations of the method",
                        required=True)
    parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                        help="Number of initialization points",
                        required=False, default=10)
    parser.add_argument("-np", dest="npts", type=int, metavar="npts",
                        help="Number of random points to sample at each iteration",
                        required=False, default=1000)
    parser.add_argument("-v", dest="verbose", action='store_true',
                        help="Verbose")
    parser.add_argument("--target", dest="target", type=str,
                        default="ZDT1",
                        help="Target function name")
    parser.add_argument("-seed", dest="seed", type=int, metavar="seed",
                        help="Seed for random number generator",
                        required=False, default=10)

    parser.set_defaults(Reduce=False)

    args = parser.parse_args()

    NParam = args.ND
    NIter = args.NI
    N_init = args.NInit
    n_pts = args.npts
    verbose = args.verbose
    seed = args.seed
    target = targets.target(args.target.lower(),NParam)
    f1 = target.f1
    f2 = target.f2
    PB = target.PB
    func = target.func

    Filename = args.target + ".dat"


    Optimize = mo.MOBayesianOpt(target=func,
                                NObj=2,
                                pbounds=PB,
                                Picture=True,
                                MetricsPS=False,
                                TPF=np.asarray([f1, f2]).T,
                                verbose=verbose,
                                Filename=Filename,
                                max_or_min='min',
                                RandomSeed=seed)

    Optimize.initialize(init_points=N_init)

    front, pop = Optimize.maximize_smsego(n_iter=NIter, n_pts=n_pts)
    PF = np.asarray([np.asarray(y) for y in Optimize.y_Pareto])
    PS = np.asarray([np.asarray(x) for x in Optimize.x_Pareto])

    FileName = "SMS-EGO_" + Filename
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
