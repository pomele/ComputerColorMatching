import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
from models.G_D import Generator, Discriminator
import data
import dataset
from dataset.ReadFileDataSet import ReadFileDataSet
from tools.AverageMeter import AverageMeter
# from IPython import embed
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating work (Generator)
KM_COMPONENTS = 15  # it could be total point G can draw in the canvas
SOURCE_COLOR_NUM = 2
ONE_D_N_G = 1
N_EPOCHS = 100
PAINT_POINTS = np.vstack([np.linspace(-1, 1, KM_COMPONENTS) for _ in range(BATCH_SIZE)])
TRAIN_DATA_FILE = 'colordata.txt'  # 改成你的数据文件的路径


def main():

    G = Generator(N_IDEAS=N_IDEAS, SOURCE_COLOR_NUM=SOURCE_COLOR_NUM)
    D = Discriminator(SOURCE_COLOR_NUM=SOURCE_COLOR_NUM)

    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = ReadFileDataSet(file_path=TRAIN_DATA_FILE)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    for i in range(N_EPOCHS):
        train(data_loader=train_loader, model_G=G, model_D=D, criterion=criterion, opt_G=opt_G, opt_D=opt_D)
    # plt.ion()  # something about continuous plotting
    # plt.ioff()
    # plt.show()
    #
    # # plot a generated painting for upper class
    # z = torch.randn(1, N_IDEAS)
    # label = torch.FloatTensor([[1.]])  # for upper class
    # G_inputs = torch.cat((z, label), 1)
    # G_paintings = G(G_inputs)
    # plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='G painting for upper class', )
    # plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3,
    #          label='upper bound (class 1)')
    # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3,
    #          label='lower bound (class 1)')
    # plt.ylim((0, 3));
    # plt.legend(loc='upper right', fontsize=10);
    # plt.show()


def train(data_loader, model_G, model_D, criterion, opt_G, opt_D):
    # 如何生成某一种颜色的配方，这个是完全随便生成，包括P目标和P颜料
    # label应该是p目标
    # colormatching应该是浓度
    model_G.train()
    model_D.train()

    D_fake_loss = AverageMeter()      # 初始化为0
    D_real_loss = AverageMeter()
    G_loss = AverageMeter()
    for i, (labels, color_match) in enumerate(data_loader):
        # color_match, labels = km_works_with_labels()  # from K-M
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
        G_inputs = torch.cat((G_ideas, labels), dim=1)  # ideas with labels size is batchsize*(31+5)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        G_matching = model_G(G_inputs)  # fake from newbie w.r.t label from G
        D_inputs1 = torch.cat((F.sigmoid(G_matching), labels), dim=1)
        prob_matching1 = model_D(D_inputs1)  # D try to reduce this prob
        fake_gt = torch.zeros(BATCH_SIZE).fill_(1).long()
        errD_fake = criterion(prob_matching1, fake_gt)

        D_inputs0 = torch.cat((color_match, labels), dim=1)  # all have their labels
        prob_matching0 = model_D(D_inputs0)  # D try to increase this prob
        real_gt = torch.zeros(BATCH_SIZE).long()
        errD_real = criterion(prob_matching0, real_gt)

        errD = errD_fake + errD_real
        opt_D.zero_grad()
        errD.backward(retain_graph=True)  # reusing computational graph
        opt_D.step()

        D_fake_loss.update(errD_fake.item(), G_inputs.size(0))
        D_real_loss.update(errD_real.item(), G_inputs.size(0))

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for i in range(ONE_D_N_G):
            opt_G.zero_grad()
            prob_matching1 = model_D(D_inputs1)  # D try to reduce this prob
            err_G = criterion(prob_matching1, real_gt)
            err_G.backward(retain_graph=True)
            opt_G.step()

            G_loss.update(err_G.item(), G_inputs.size(0))

        print('D_fake_loss=%.3f(%.3f) \t D_real_loss=%.3f(%.3f) \t G_loss=%.3f(%.3f)' % (
        D_fake_loss.val, D_fake_loss.avg, D_real_loss.val, D_real_loss.avg, G_loss.val, G_loss.avg))

        # if step % 50 == 0:  # plotting
        #     plt.cla()
        #     plt.plot(PAINT_POINTS[0], G_matching.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        #     bound = [0, 0.5] if labels.data[0, 0] == 0 else [0.5, 1]
        #     plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3,
        #              label='upper bound')
        #     plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3,
        #              label='lower bound')
        #     plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_matching0.data.numpy().mean(),
        #              fontdict={'size': 13})
        #     plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        #     plt.text(-.5, 1.7, 'Class = %i' % int(labels.data[0, 0]), fontdict={'size': 13})
        #     plt.ylim((0, 3));
        #     plt.legend(loc='upper right', fontsize=10);
        #     plt.draw();
        #     plt.pause(0.1)


if __name__ == "__main__":
    # km_works_with_labels()
    main()
