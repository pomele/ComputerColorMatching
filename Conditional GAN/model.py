import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
KM_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, KM_COMPONENTS) for _ in range(BATCH_SIZE)])



def km_works_with_labels():     # K-M data (real target)

    optical_model = 'km'  # the optical model to use
    sigma = 0.2  # the noise std
    ydim = 31  # number of data samples
    bound = [0., 1., 0., 1.]  # effective bound for likelihood
    seed = 1  # seed for generating data

    # generate data
    concentrations, reflectance, x = data.generate(
        model=optical_model,
        tot_dataset_size=2 ** 10,
        ydim=ydim,
        sigma=sigma,
        prior_bound=bound,
        seed=seed
    )
    colormatching = torch.cat(concentrations, reflectance, x)
    # label  wrong
    label = torch.from_numpy(colormatching.astype(np.float32))
    return colormatching, label


G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS+1, 128),          # random ideas (could from normal distribution) + class label
    nn.ReLU(),
    nn.Linear(128, KM_COMPONENTS),     # making a color matching from these random ideas
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(KM_COMPONENTS+1, 128),   # receive art work either from the K-M or a newbie like G with label
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the matching is made by K-M
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()   # something about continuous plotting

# 如何生成某一种颜色的配方，这个是完全随便生成，包括P目标和P颜料
# label应该是p目标
# colormatching应该是浓度
for step in range(10000):
    colormatching, labels = km_works_with_labels()           # from K-M
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)               # random ideas
    G_inputs = torch.cat((G_ideas, labels), 1)               # ideas with labels
    G_matching = G(G_inputs)                                 # fake from newbie w.r.t label from G

    D_inputs0 = torch.cat((colormatching, labels), 1)        # all have their labels
    D_inputs1 = torch.cat((G_matching, labels), 1)
    prob_matching0 = D(D_inputs0)                 # D try to increase this prob
    prob_matching1 = D(D_inputs1)                 # D try to reduce this prob

    D_score0 = torch.log(prob_matching0)          # maximise this for D
    D_score1 = torch.log(1. - prob_matching1)     # maximise this for D
    D_loss = - torch.mean(D_score0 + D_score1)  # minimise the negative of both two above for D
    G_loss = torch.mean(D_score1)               # minimise D score w.r.t G

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_matching.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        bound = [0, 0.5] if labels.data[0, 0] == 0 else [0.5, 1]
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_matching0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.text(-.5, 1.7, 'Class = %i' % int(labels.data[0, 0]), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.1)

plt.ioff()
plt.show()

# plot a generated painting for upper class
z = torch.randn(1, N_IDEAS)
label = torch.FloatTensor([[1.]])     # for upper class
G_inputs = torch.cat((z, label), 1)
G_paintings = G(G_inputs)
plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='G painting for upper class',)
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 1)')
plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.show()

