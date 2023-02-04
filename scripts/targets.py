import deap.benchmarks as db
import numpy as np

def zdt1(x):
    return np.asarray(db.zdt1(x))

def zdt2(x):
    return np.asarray(db.zdt2(x))

def zdt3(x):
    return np.asarray(db.zdt3(x))

def schaffer_mo(x):
    return np.asarray(db.schaffer_mo(x))

def fonseca(x):
    return np.asarray(db.fonseca(x))