import argparse
import logging

import numpy as np
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.manifolds.poincareball import PoincareBall

import hyptorch.pmath as pmath
import hyptorch.nn as pnn


from htsne_impl import TSNE as hTSNE
from sklearn.manifold import TSNE


c = 1.0

def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = torch.normal(mean=0.0, std=1.0, size=(dimension, num_points))
    random_directions /= torch.norm(random_directions, dim=0)
    
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = torch.rand(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def generate_riemannian_distri(idx, batch=10, dim=2, scale=1., all_loc=[]):
    
    pball = PoincareBall(dim, c=1)

    loc = random_ball(1, dim, radius=0.999)

    if idx == 0:
        loc = torch.zeros_like(loc)

    distri = RiemannianNormal(loc, torch.ones((1,1)) * scale, pball)

    return distri, loc


def generate_riemannian_clusters(clusters=5, batch=20, dim=2, scale=1.):
    embs  = torch.zeros((0, dim))
    means = torch.zeros((0, dim))
    
    pball = PoincareBall(dim, c=1)


    all_loc = []

    labels= []
    

    for i in range(clusters):

        distri, mean = generate_riemannian_distri(idx = i, batch=batch, dim=dim, scale=scale, all_loc=all_loc)

        labels.extend([i] * batch)

        for _ in range(batch):
            embs = torch.cat((embs, distri.sample()[0]))

        means = torch.cat((means, mean))

    ###############################################

    return embs, labels, means


def generate_high_dims():

    embs, labels, means = generate_riemannian_clusters(clusters=5, batch=20, dim=5, scale=0.25)

    print("embs", embs.shape)
    
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    seed_colors = ['black', 'red', 'b', 'g', 'c']

    colors = []
    for label in labels:
        colors.append(seed_colors[label])

    plt.scatter(embs[:,0], embs[:,1], c=colors, alpha=0.3)


    mcolors = []
    for i in range(means.shape[0]):
        mcolors.append(seed_colors[i])

    plt.scatter(means[:,0], means[:,1], c=mcolors, marker='x', s=50)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)
    #####################################################


    plt.savefig("./saved_figures/high_dim" + ".png", bbox_inches='tight', dpi=fig.dpi)

    return embs, colors



def run_TSNE(embeddings, learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0):

    tsne = TSNE(n_components=2, method='exact', perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=1)

    tsne_embeddings = tsne.fit_transform(embeddings)

    print ("\n\n")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()


    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)


    _htsne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=0.0, student_t_gamma=1.0, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    HT_SNE_embeddings = _htsne.fit_transform(dists, embeddings)


    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding



def plot_low_dims(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma):



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/" + "tsne.eps", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/" + "HT-SNE.eps", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=30)
    ax.set_aspect('equal')
    plt.axis('off')

    plt.savefig("./saved_figures/" + "CO-SNE.eps", bbox_inches='tight', dpi=fig.dpi)



if __name__ == "__main__":

    embeddings, colors = generate_high_dims()


    learning_rate = 5.0
    learning_rate_for_h_loss = 0.1
    perplexity = 20
    early_exaggeration = 1.0
    student_t_gamma = 0.1


    tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding  = run_TSNE(embeddings, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)

    plot_low_dims(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)


