import deap.benchmarks as db
import numpy as np


class target:
    def __init__(self,name, Nparam=1):
        self.name = name
        if self.name == 'zdt1':
            self.f1 = np.linspace(0, 1, 1000)
            self.f2 = 1 - np.sqrt(self.f1)
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = zdt1
        elif self.name == 'zdt2':
            self.f1 = np.linspace(0, 1, 1000)
            self.f2 = 1 - self.f1 ** 2
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = zdt2
        elif self.name == 'zdt3':
            f1 = np.linspace(0, .08300, 200)
            f1 = np.append(f1, np.linspace(.1822, .25770, 200))
            f1 = np.append(f1, np.linspace(.4093, .45380, 200))
            f1 = np.append(f1, np.linspace(.6183, .65250, 200))
            f1 = np.append(f1, np.linspace(.8233, .8518, 200))
            self.f1 = np.append(f1, np.linspace(.9450, 1, 200))
            self.f2 = 1 - np.sqrt(self.f1) - self.f1 * np.sin(10 * np.pi * self.f1)
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = zdt3
        elif self.name == 'zdt4':
            self.f1 = np.linspace(0, 1, 1000)
            self.f2 = 1 - np.sqrt(self.f1)
            PB = np.append(np.asarray([[0, 1]]), [[-10, 10]] * (Nparam - 1))
            self.PB = np.reshape(PB, [Nparam, 2])
            self.func = zdt4
        elif self.name == 'zdt6':
            x = np.linspace(0, 1, 1000)
            self.f1 = 1 - np.exp(-4*x) * np.sin(6*np.pi*x)**6
            self.f2 = 1 - self.f1**2
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = zdt6
        elif self.name == 'schaffer':
            x = np.linspace(0, 2, 100)
            self.f1 = (x ** 2 )
            self.f2 = (x - 2) ** 2
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = schaffer_mo
        elif self.name == 'fonseca':
            x = np.linspace(0, 1, 10000)
            self.f1 = 1 - np.exp(-3 * ((8 * x - 4 - 1 / np.sqrt(3)) ** 2))
            self.f2 = 1 - np.exp(-3 * ((8 * x - 4 + 1 / np.sqrt(3)) ** 2))
            self.PB = np.asarray([[0, 1]] * Nparam)
            self.func = fonseca
        else:
            raise ValueError('Invalid target function')
def zdt1(x):
    return np.asarray(db.zdt1(x))

def zdt2(x):
    return np.asarray(db.zdt2(x))

def zdt3(x):
    return np.asarray(db.zdt3(x))

def zdt4(x):
    return np.asarray(db.zdt4(x))

def zdt6(x):
    return np.asarray(db.zdt6(x))

def schaffer_mo(x):
    return np.asarray(db.schaffer_mo(x))

def fonseca(x):
    return np.asarray(db.fonseca(x))