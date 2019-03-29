# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:37:22 2019

@author: FaiHuntrakool
"""

import networkx as nx
G=nx.social.florentine_families_graph()
nx.draw_networkx(G,with_labels=True,node_color='y')

import dwave_networkx as dnx
import dimod
sampler=dimod.ExactSolver()
response=dnx.maximum_cut(G,sampler)
#print(response)

def find_min_set_max_cut(G,num_repeat,sampler):
    if sampler=='SimulatedAnnealing':
        import neal
        sampler=neal.SimulatedAnnealingSampler()
    else:
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite
        sampler=EmbeddingComposite(DWaveSampler())
    from collections import defaultdict
    min_set=defaultdict(list)
    for i in range(num_repeat):
        response=dnx.maximum_cut(G,sampler)
        min_set[len(response)].append(list(response))
    return min(min_set,key=min_set.get)
        

ans=find_min_set_max_cut(G,10,'SimulatedAnnealing')
print('len_max_cut='+str(ans))
print('exact solver len='+str(len(response)))