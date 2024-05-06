#! /usr/bin/env python3
import sys
import numpy as np
from scipy.stats import pointbiserialr

####################
# Priority Queue

_PQ_t = list[tuple[{'__eq__', }, {'__lt__', '__ge__', '__eq__'}]]
def PQ_new() -> _PQ_t:
    return []

def PQ_push(queue: _PQ_t, item, prio): 
    """
    item already in priority queue with smaller priority:
    -> update its priority
    item already in priority queue with higher priority:
    -> do nothing
    if item not in priority queue:
    -> push it
    """
    # naive approach, but should work fine on low elements
    for idx, (it, pri) in enumerate(queue): # if item in list
        if it == item: 
            if pri >= priority: # low priority, no update
                return
            del s.q[idx] # high priority, update
            queue.append((item, prio))
            return
    queue.append((item, prio)) # item not in list

def PQ_pop(queue: _PQ_t) -> tuple['item', 'priority']:
    'return item with highest priority and remove it from queue'
    top_priority, top_index = queue[0][1], 0
    for index, (_, p) in enumerate(queue):
        if (top_priority < p):
            top_priority, top_index = p, index
    return queue.pop(top_index)

def test_PQ():  
    q = PQ_new()
    PQ_push(q, 1, 0.5)
    PQ_push(q, 2, 0.2)
    PQ_push(q, 3, 1.0)
    assert (3, 1.0) == PQ_pop(q)
    assert (1, 0.5) == PQ_pop(q)
    assert (2, 0.2) == PQ_pop(q)


##############################
# Decorrelation

def get_merit(df: 'pd.DataFrame', features: list, cls_lbl: list):
    k = len(features)
    # average feature-class abs correlation
    feature_cls_corr = 0 
    for cls in cls_lbl:
        for f in features:
            corr, __pval = pointbiserialr(
                df[f].values, df[cls].values
            )
            feature_cls_corr += abs(corr)
    if k == 1: # special short case
        return feature_cls_corr
    feature_cls_corr /= k
    # feature-feature abs correlation
    corr_mat = df[features].corr().values
    corr_mat[np.tril_indices_from(corr_mat)] = np.nan
    # ^^^ remove redundant values
    corr_mat = np.abs(corr_mat)
    feature_feature_corr = np.nanmean(np.abs(corr_mat))
    # compute merit
    rfc = feature_cls_corr
    rff = feature_feature_corr
    merit = (k * rfc) / np.sqrt(k+ k*(k-1)*rff)
    return merit


class CorrelationBasedFeatureSelect:
    """
    select uncorrelated feature columns from dataframe.
    NOTE: takes long time (~2 min) 
    on medim datasets (~500 features x 1000 cases)
    (due to inefficient implementation)
    usage: 
	```
	r = CBSF()
	# start = time.time()
	r(df, features=[1, 2, 3], cls_lbl=0)
	# stop = time.time()
	# print(stop - start)

	print(r.best_merit)
	print(list(r.best_feature))
	```
	links:
    # https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/
    # https://researchcommons.waikato.ac.nz/bitstream/handle/10289/1024/uow-cs-wp-2000-08.pdf?sequence=1&isAllowed=y
    """
    __slots__ = ("best_merit", "best_feature", "queue")
    
    def __call__(
        result, 
        df: 'pd.DataFrame', 
        features: list, 
        cls_lbl: list, 
        max_backtracks: int = 5,
    ):
        # sometimes my mind comes up with strange class-related solutions
        # here i use `self` argument as a mutable shared state
        
        # coerce to list
        cls_lbl = list(cls_lbl) if hasattr(cls_lbl, "__iter__") else [cls_lbl, ]
        # zeroth iteration
        result.best_merit = -1 
        result.best_feature = None
        for feature in features:
            merit = get_merit(df, [feature, ], cls_lbl)
            if merit > result.best_merit:
                best_merit, best_feature = merit, feature

        result.best_feature = {best_feature, }
        result.best_merit = best_merit
        
        # rest of iterations
        queue = PQ_new()
        result.queue = queue
        PQ_push(queue, result.best_feature, result.best_merit)
        visited = []
        n_backtracks = 0
        # iteration = 0
        while len(queue):
            # iteration +=1  # maybe add progress callback? 
            # if not iteration%10:
                # print(iteration)
            feature_sub, sub_merit = PQ_pop(queue)

            if sub_merit < result.best_merit:
                n_backtracks +=1
            else:
                result.best_merit = sub_merit
                result.best_feature = feature_sub

            if (n_backtracks >= max_backtracks):
                break

            for feature in features:
                new_sub = feature_sub.copy()
                new_sub.add(feature)
                for v in visited:
                    if v == new_sub:
                        break
                else:
                    visited.append(new_sub)
                    merit = get_merit(df, list(new_sub), cls_lbl)
                    PQ_push(queue, new_sub, merit)
        return result
    
CBFS = CorrelationBasedFeatureSelect

