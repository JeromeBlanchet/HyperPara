"""
    @file:              HpDispatcher.py
    @Author:            Gabriel Gibeau Sanchez
    @Creation Date:     04/08/2020

    @Description:       This file provide a class that manages the multiprocess dispatching of the
                        hyperparameters search methods. It leverages the functionnalities of the
                        multiprocessing package
"""

import multiprocessing as mp
from enum import Enum, unique
import numpy as np
import datetime as dt
import pandas as pd


class HpDispatcher:
    """
    Class that manages the parallel dispatching for hyperparameters search methods
    """
    # def __call__(self, *args, **kwargs):
    #     pass

    def __init__(self):
        # initialize pool with avalaible cpu cores
        self.pool = mp.Pool(mp.cpu_count())

    def imap_iterable(self, iterable_func, args, chunksize=10):
        res = self.pool.imap(iterable_func, args, chunksize=chunksize)

        return res