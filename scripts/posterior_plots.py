#Script to plot the prior and posterior distributions of the Gaussian process
import targets
import mobopt as mo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="ND", type=int, metavar="ND",
                    help="Number of Dimensions",
                    default=30,
                    required=False)
parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                    help="Number of initialization points",
                    required=False, default=10)
parser.add_argument("-np", dest="npts", type=int, metavar="npts",
                    help="Number of random points to sample at each iteration",
                    required=False, default=20)
parser.add_argument("--target", dest="target", type=str,
                    default="ZDT1",
                    help="Target function name")
parser.add_argument("-neval", dest="neval", type=int, metavar="neval",
                    help="Number of random points for GP evaluation",
                    required=False, default=20)
parser.add_argument("-seed", dest="seed", type=int, metavar="seed",
                    help="Seed for random number generator",
                    required=False, default=10)

parser.set_defaults(Reduce=False)

args = parser.parse_args()

NParam = args.ND
N_init = args.NInit
n_pts = args.npts
neval = args.neval
seed = args.seed
target = targets.target(args.target.lower(), NParam)
f1 = target.f1
f2 = target.f2
PB = target.PB
func = target.func

iterations = [10, 30, 50]

for NIter in iterations:

    Optimize = mo.MOBayesianOpt(target=func,
                                NObj=2,
                                pbounds=PB,
                                MetricsPS = False,
                                max_or_min='min',
                                RandomSeed=seed)

    Optimize.initialize(init_points=N_init)

    if NIter == 10:
        Optimize.space.plot_gp(gpr_model=Optimize.GP, n_eval_pts=neval, title=args.target.lower()+"_prior", seed=seed+NIter)

    front, pop = Optimize.maximize_smsego(n_iter=NIter, n_pts=n_pts)

    title = args.target.lower() + "_iter=" + str(NIter) + "_posterior"
    Optimize.space.plot_gp(gpr_model=Optimize.GP, n_eval_pts=neval, title=title, seed=seed+NIter)

