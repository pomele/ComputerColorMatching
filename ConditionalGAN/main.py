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
TRAIN_DATA_FILE = 'colordata.txt'  # 训练数据文件的路径
VAL_DATA_FILE = 'valcolordata.txt'  # 验证集数据文件的路径

VALIDATE = 1

def main():

    G = Generator(N_IDEAS=N_IDEAS, SOURCE_COLOR_NUM=SOURCE_COLOR_NUM)
    D = Discriminator(SOURCE_COLOR_NUM=SOURCE_COLOR_NUM)

    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = ReadFileDataSet(file_path=TRAIN_DATA_FILE)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # val_dataset = ReadFileDataSet(file_path=VAL_DATA_FILE)
    for i in range(N_EPOCHS):
        train(data_loader=train_loader, model_G=G, model_D=D, criterion=criterion, opt_G=opt_G, opt_D=opt_D)
        # if VALIDATE:
        #     validate(data_loader=val_loader, model_G=G, model_D=D, criterion=criterion, opt_G=opt_G, opt_D=opt_D)


def train(data_loader, model_G, model_D, criterion, opt_G, opt_D):
    # label应该是p目标
    # colormatching应该是浓度
    model_G.train()
    model_D.train()

    D_fake_loss = AverageMeter()      # 初始化为0
    D_real_loss = AverageMeter()
    G_loss = AverageMeter()
    for i, (labels, color_match) in enumerate(data_loader):

        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)      # G_ideas is noise , random ideas
        G_inputs = torch.cat((G_ideas, labels), dim=1)  # ideas with labels size is batchsize*(31+5)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        G_matching = model_G(G_inputs)  # fake from newbie w.r.t label from G
        D_inputs1 = torch.cat((torch.sigmoid(G_matching), labels), dim=1)
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


        cal_diff(G_matching.detach().numpy(), labels.detach().numpy())


def validate(val_loader, model_G, model_D, criterion, opt_G, opt_D):
    print('validate')





def cal_diff(concentration, reflectance):
    # the first data -> plot
    # data -> reflectance
    # plot
    print(concentration.shape)
    print(reflectance.shape)
    Ngrid = 64
    optical_model = "km"
    sigma = 0.2  # the noise std
    bound = [0., 1., 0., 1.]  # effective bound for likelihood

    temp = data.get_lik(concentration, reflectance, n_grid=Ngrid, model=optical_model,
                        sigma=sigma, xvec=None, bound=bound)
    print('++++++++++++++++++++++色差++++++++++++++++++++++++++++++')
    print(temp)
    print('----------------------色差------------------------------')


if __name__ == "__main__":
    # km_works_with_labels()
    main()
