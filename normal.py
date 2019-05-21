# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:13:01 2019

@author: FaiHuntrakool
"""

import pandas as pd
import numpy as np

n_cov=2
sample=1000
n_bin=10
mat=np.random.normal(1,2,(sample,n_cov))
solver='SA'

def create_bins_array(initial_mat,no_bins,n_cov):
    bmat=np.zeros((len(initial_mat),len(initial_mat[0])))
    for b in range(n_cov):
        data=initial_mat[:,b]
        bins=np.linspace(data.min(),data.max(),no_bins)
        digitized=np.digitize(data,bins)
        bmat[:,b]=digitized
    return bmat
bins=create_bins_array(mat,n_bin,n_cov)


import matplotlib.pyplot as plt
for i in range(n_cov):
    cov=pd.DataFrame(bins[:,i]).rename({0:'bins'},axis=1)
    plt.title('Covariate '+str(i+1))
    plt.hist(list(cov['bins']),density=False,histtype='bar',color='black')
    plt.xticks(range(0,n_bin))
    plt.show()

###########model formulation ################################
"""check duayyy"""
from pyqubo import Array
s=Array.create('s',shape=sample,vartype='SPIN')
def objective_function(s,bins,n_bin):
    H=0
    for col in range(len(mat[0])):
        for b in range(1,n_bin+1):
            M=0
            for val,ss in zip(bins[:,col],s):
                if val==b:
                    M=M+ss
            H=H+(M-0)**2
    return H-0
H=objective_function(s,bins,n_bin)
model=H.compile()
qubo,offset=model.to_qubo()
bqm=model.to_dimod_bqm()###check
################# show qubo matrix ##########################
import re
qmat=np.zeros((sample,sample))
for key,val in qubo.items():
    x=int(int(re.search(r'\d+', key[0]).group()))
    y=int(int(re.search(r'\d+', key[1]).group()))
    qmat[min(x,y)][max(x,y)]=val

import pandas as pd
q=pd.DataFrame(qmat)
#############################################################
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import neal
import timeit
start = timeit.default_timer()
if solver=='Exact':
    import dimod
    sampler=dimod.ExactSolver()
    response=sampler.sample(bqm)
else:
    #sampler=EmbeddingComposite(DWaveSampler())
    sampler=neal.SimulatedAnnealingSampler()    
    response=sampler.sample(bqm,num_reads=100)
stop = timeit.default_timer()

print('time:= '+str(stop-start)+'s')
###########Embedding visualization #########################

import matplotlib.pyplot as plt
for i in range(n_cov):
    cov=pd.DataFrame(bins[:,i]).rename({0:'bins'},axis=1)
    plt.title('Covariate '+str(i+1))
    plt.hist(list(cov['bins']),density=False,histtype='bar',color='black')
    plt.xticks(range(0,n_bin))
    plt.show()
"""evaluation"""
###########result####################################
from collections import defaultdict    
result=defaultdict(list)
for r in response.data(['sample', 'energy']):
    result[r.energy].append(r.sample)

res=pd.DataFrame()
min_energy=min(result.keys())
x=result[min_energy]
res=pd.DataFrame.from_dict(x[0],'index').transpose()
###########visualization####################################
group=pd.DataFrame(res.iloc[0,:]).reset_index().drop('index',axis=1).rename({0:'group'},axis=1)

colors=['red','blue']
for i in range(n_cov):
    cov=pd.DataFrame(bins[:,i]).rename({0:'bins'},axis=1)
    group=pd.DataFrame(res.iloc[0,:]).reset_index().drop('index',axis=1).rename({0:'group'},axis=1)
    df=pd.concat([cov, group], axis=1)
    plt.title('Covariate '+str(i+1))
    plt.hist([list(df[df['group']==0]['bins']),list(df[df['group']==1]['bins'])],density=False,histtype='bar',color=colors,bins=n_bin)
    plt.show()

##########Variance, mean (Compare with randomization technique)##################################
dcov=dict()
for i in range(n_cov):
    cov=pd.DataFrame(mat[:,i]).rename({0:'bins'},axis=1)
    df=pd.concat([cov, group], axis=1)
    dcov[i]={'mean':df['bins'].mean(),
            'variance':df['bins'].var(),
            'mean_control':df[df['group']==0]['bins'].mean(),
            'variance_control':df[df['group']==0]['bins'].var(),
            'mean_treatment':df[df['group']==1]['bins'].mean(),
            'variance_treatment':df[df['group']==1]['bins'].var()}
#randomization
group2=pd.DataFrame(np.random.randint(0,2,(sample,1))).rename({0:'group'},axis=1)


colors=['red','blue']
for i in range(n_cov):
    cov=pd.DataFrame(bins[:,i]).rename({0:'bins'},axis=1)
    df=pd.concat([cov, group2], axis=1)
    plt.title('Covariate '+str(i+1))
    plt.hist([list(df[df['group']==0]['bins']),list(df[df['group']==1]['bins'])],density=False,histtype='bar',color=colors,bins=n_bin)
    plt.show()


dcovrand=dict()
for i in range(n_cov):
    cov=pd.DataFrame(mat[:,i]).rename({0:'bins'},axis=1)
    df=pd.concat([cov, group2], axis=1)
    dcovrand[i]={'mean':df['bins'].mean(),
            'variance':df['bins'].var(),
            'mean_control':df[df['group']==0]['bins'].mean(),
            'variance_control':df[df['group']==0]['bins'].var(),
            'mean_treatment':df[df['group']==1]['bins'].mean(),
            'variance_treatment':df[df['group']==1]['bins'].var()}
#print(pd.DataFrame.from_dict(dcovrand))

df=pd.DataFrame.from_dict(dcov).transpose()
df['different of means-optimization']=(df['mean_treatment']-df['mean_control']).abs()
df_rand=pd.DataFrame.from_dict(dcovrand).transpose()
df_rand['different of means-before optimization']=(df_rand['mean_treatment']-df_rand['mean_control']).abs()
result2 = pd.concat([df['different of means-optimization'], df_rand['different of means-before optimization']], axis=1, sort=False)

print(result2.mean())
print('solution')
print(min(result.keys()))
