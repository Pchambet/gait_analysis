#%% main.py
import numpy as np
import functions.extract_data as ed
import functions.show as show
import functions.clustering as clus
import functions.DTW as dtw
import functions.methods as meth
import functions.ACP as acp

X = ed.phases("Norm_V1.mat")
show.show_curve(X, title = "knee flexion")

meth.view_3D()

#meth.variance()


# %%
