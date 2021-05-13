#%%
import random

import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree

class point():

    def __init__(self,label,x,y):
        self.name = label
        self.x = x
        self.y = y
        self.pos = (x,y)

def plot_tree(points, edges):

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    for p in points:
        ax.scatter(p.x,p.y,s=20,marker="s",edgecolor="k",color="k")
        ax.annotate(p.name,p.pos,ha="center",va="top")

    for edge in edges:
        p0 = [p for p in points if p.name == edge[0]][0]
        p1 = [p for p in points if p.name == edge[1]][0]
        ax.plot([p0.x,p1.x],[p0.y,p1.y],zorder=0)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

def random_points(n_points,xmin,xmax,ymin,ymax):

    labels = ["B_"+str(pn) for pn in range(n_points)]

    points = [point(lab,random.randint(xmin,xmax),random.randint(ymin,ymax)) for lab in labels]

    labels_d = {pn:"B_"+str(pn) for pn in range(n_points)}

    return points, labels_d

def distance(p0,p1):
    return ((p1.y - p0.y)**2 + (p1.x - p0.x)**2)**0.5

def distance_matrix(points):
    return np.array([[distance(p0,p1) for p1 in points] for p0 in points])

#%% TEST

n_points = 10
xmin = 0
xmax = 5000
ymin = 0
ymax = 5000

points, labels_d = random_points(10,0,5000,0,5000)

d_matrix = distance_matrix(points)

tree = minimum_spanning_tree(d_matrix)

edges = [(labels_d[k[0]],labels_d[k[1]]) for k in tree.todok().keys()]

plot_tree(points, edges)

print("Total distance:",int(np.sum(tree))," m")

# %%
new_point = point("P_"+str(len(points)+1),random.randint(xmin,xmax),random.randint(ymin,ymax))

# Compute distance to all other points and choose shortest edge
np_distances = {(new_point.name,p.name):distance(new_point,p) for p in points}

new_edge = min(np_distances, key=np_distances.get)

points.append(new_point)
edges.append(new_edge)

plot_tree(points,edges)