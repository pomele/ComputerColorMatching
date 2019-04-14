import torch
import torch.nn as nn
import numpy as np
import os
# import matplotlib.pyplot as plt
from models.G_D import Generator, Discriminator
import data
import dataset
from dataset.ReadFileDataSet import ReadFileDataSet
from tools.AverageMeter import AverageMeter
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from IPython import embed
from tools.plot_figure import plot_curves
from tools.plot_figure import plot_diff

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 1  # think of this as number of ideas for generating work (Generator)
KM_COMPONENTS = 15  # it could be total point G can draw in the canvas
SOURCE_COLOR_NUM = 6
ONE_D_N_G = 3
N_EPOCHS = 10
PAINT_POINTS = np.vstack([np.linspace(-1, 1, KM_COMPONENTS) for _ in range(BATCH_SIZE)])
TRAIN_DATA_FILE = 'colordata.txt'  # 训练数据文件的路径
VAL_DATA_FILE = 'valcolordata.txt'  # 验证集数据文件的路径

VALIDATE = True

MODEL_PATH = 'models/'


def main():
    G = Generator(N_IDEAS=N_IDEAS, SOURCE_COLOR_NUM=SOURCE_COLOR_NUM).to(device)
    D = Discriminator(SOURCE_COLOR_NUM=SOURCE_COLOR_NUM).to(device)

    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

    criterion = nn.CrossEntropyLoss().to(device)
    mse_criterion = nn.MSELoss().to(device)

    train_dataset = ReadFileDataSet(file_path=TRAIN_DATA_FILE)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    val_dataset = ReadFileDataSet(file_path=VAL_DATA_FILE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    train_D_fake_losses = []
    train_D_real_losses = []
    train_G_losses = []
    val_D_fake_losses = []
    val_D_real_losses = []
    val_G_losses = []
    difference = []

    pretrain_G_first = True
    checkpoint_path = os.path.join(MODEL_PATH, 'pre_trained_G.pkl')
    TRAIN_G_FIRST = 2
    best_G_loss = 1000000
    for epoch in range(N_EPOCHS):
        if epoch == 0:
            if pretrain_G_first:
                for x in range(TRAIN_G_FIRST):
                    pretrain_g_loss = train_G_only(data_loader=val_loader, model_G=G, mse_criterion=mse_criterion,
                                                   opt_G=opt_G, epoch=epoch)
                    # 这里可以选择保存G的参数(torch.save())
                    save_checkpoint({
                        'epoch': x,
                        'state_dict': G.state_dict(),
                        'val_result': pretrain_g_loss,
                        'optimizer': opt_G.state_dict(),
                    }, is_best=True, checkpoint_dir=MODEL_PATH, checkpoint_name='pre_trained_G.pkl',
                        info='save pretrained G model state')

            elif os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                G.load_state_dict(checkpoint['state_dict'])
                print('load G state successfully')
            else:
                raise ValueError('the checkpoint path is invalid.....')

        train_D_fake_loss, train_D_real_loss, train_G_loss = train(data_loader=train_loader, model_G=G, model_D=D,
                                                                   criterion=criterion, opt_G=opt_G, opt_D=opt_D,
                                                                   epoch=epoch)
        train_D_fake_losses.append(train_D_fake_loss)
        train_D_real_losses.append(train_D_real_loss)
        train_G_losses.append(train_G_loss)
        # evaluate on validation set
        if VALIDATE:
            val_D_fake_loss, val_D_real_loss, val_G_loss, epoch_diff = validate(data_loader=val_loader, model_G=G, model_D=D,
                                                                    criterion=criterion, epoch=epoch)
            val_D_fake_losses.append(val_D_fake_loss)
            val_D_real_losses.append(val_D_real_loss)
            val_G_losses.append(val_G_loss)
            difference.append(epoch_diff)

            # save checkpoint when the G loss is the best
            if val_G_loss < best_G_loss:
                best_G_loss = min(best_G_loss, val_G_loss)
                save_checkpoint_info = 'epoch:%d \t avg_D_fake_loss=%.3f \t avg_D_real_loss=%.3f \t avg_G_loss=%.3f' % (
                    epoch, val_D_fake_loss, val_D_real_loss, val_G_loss)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': G.state_dict(),
                    'val_result': val_G_loss,
                    'optimizer': opt_G.state_dict(),
                }, is_best=True, checkpoint_dir=MODEL_PATH, checkpoint_name='best_G.pkl', info=save_checkpoint_info)

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': D.state_dict(),
                    'val_result': val_G_loss,
                    'optimizer': opt_D.state_dict(),
                }, is_best=True, checkpoint_dir=MODEL_PATH, checkpoint_name='best_D.pkl', info=save_checkpoint_info)
    train_data = {'D_fake_loss': train_D_fake_losses,
                'D_real_loss': train_D_real_losses,
                'G_loss': train_G_losses}
    val_data = {'D_fake_loss': val_D_fake_losses,
                'D_real_loss': val_D_real_losses,
                'G_loss': val_G_losses}
    diff_data = {'difference': difference}
    plot_curves(all_data=train_data, label='train')
    plot_curves(all_data=val_data, label='validate')
    plot_diff(all_data=diff_data)


def train_G_only(data_loader, model_G, mse_criterion, opt_G, epoch):
    model_G.train()
    G_loss = AverageMeter()

    for i, (labels, color_match) in enumerate(data_loader):
        color_match = color_match.to(device)
        ideas_randn = torch.Tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        noise_index = torch.randint(0, 5, (BATCH_SIZE,))
        cur_noise = torch.index_select(ideas_randn, 0, noise_index)
        G_ideas = cur_noise.unsqueeze(1)

        G_inputs = torch.cat((G_ideas, labels), dim=1).to(device)  # ideas with labels size is batchsize*(31+1)
        G_matching = model_G(G_inputs)  # fake from newbie w.r.t label from G
        loss = mse_criterion.forward(G_matching, color_match)
        opt_G.zero_grad()
        loss.backward()
        opt_G.step()

        G_loss.update(loss.item(), G_inputs.size(0))
        # 加一些记录loss的
        # 然后返回一个loss的平均值什么的
        print('train  train_G_only  :    ' + 'G_loss=%.3f(%.3f)' % (G_loss.val, G_loss.avg))

    return G_loss.avg


def train(data_loader, model_G, model_D, criterion, opt_G, opt_D, epoch):
    # label应该是p目标
    # colormatching应该是浓度
    model_G.train()
    model_D.train()

    D_fake_loss = AverageMeter()  # 初始化为0
    D_real_loss = AverageMeter()
    G_loss = AverageMeter()
    print('epoch %d begin the opt_G lr=%.6f opt_D lr=%.6f' % (
        epoch, opt_G.param_groups[0]['lr'], opt_D.param_groups[0]['lr']))

    batch_count = len(data_loader)
    for batch_i, (labels, color_match) in enumerate(data_loader):
        # G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)      # G_ideas is noise , random ideas
        labels = labels.to(device)
        color_match = color_match.to(device)
        ideas_randn = torch.Tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        noise_index = torch.randint(0, 5, (BATCH_SIZE,))
        cur_noise = torch.index_select(ideas_randn, 0, noise_index)
        G_ideas = cur_noise.unsqueeze(1).to(device)

        G_inputs = torch.cat((G_ideas, labels), dim=1).to(device)  # ideas with labels size is batchsize*(31+1)

        ###########################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        G_matching = model_G(G_inputs)  # fake from newbie w.r.t label from G
        D_inputs1 = torch.cat((torch.sigmoid(G_matching), labels), dim=1).to(device)
        prob_matching1 = model_D(D_inputs1)  # D try to reduce this prob
        fake_gt = torch.zeros(BATCH_SIZE).long().to(device)
        errD_fake = criterion(prob_matching1, fake_gt)

        D_inputs0 = torch.cat((color_match, labels), dim=1).to(device)  # all have their labels
        prob_matching0 = model_D(D_inputs0)  # D try to increase this prob
        real_gt = torch.zeros(BATCH_SIZE).fill_(1).long().to(device)
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

        print('train:[%d/%d]\t D_fake_loss=%.3f(%.3f) \t D_real_loss=%.3f(%.3f) \t G_loss=%.3f(%.3f)' % (
        batch_i, batch_count,
        D_fake_loss.val, D_fake_loss.avg, D_real_loss.val, D_real_loss.avg, G_loss.val, G_loss.avg))

    print('epoch:%d \t avg_D_fake_loss=%.3f \t avg_D_real_loss=%.3f \t avg_G_loss=%.3f' % (
        epoch, D_fake_loss.avg, D_real_loss.avg, G_loss.avg))

    return D_fake_loss.avg, D_real_loss.avg, G_loss.avg


def validate(data_loader, model_G, model_D, criterion, epoch):
    # switch to evaluate mode
    model_G.eval()
    model_D.eval()

    D_fake_loss = AverageMeter()  # 初始化为0
    D_real_loss = AverageMeter()
    G_loss = AverageMeter()

    D_fake_epoch = []
    D_real_epoch = []
    G_loss_epoch = []
    diff = AverageMeter()

    with torch.no_grad():
        batch_count = len(data_loader)
        for batch_i, (labels, color_match) in enumerate(data_loader):
            labels = labels.to(device)
            color_match = color_match.to(device)
            G_ideas = torch.randn(BATCH_SIZE, N_IDEAS).to(device)  # G_ideas is noise , random ideas
            G_inputs = torch.cat((G_ideas, labels), dim=1)  # ideas with labels size is batchsize*(31+5)

            ###########################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            G_matching = model_G(G_inputs)  # fake from newbie w.r.t label from G
            D_inputs1 = torch.cat((torch.sigmoid(G_matching), labels), dim=1)
            prob_matching1 = model_D(D_inputs1)  # D try to reduce this prob
            fake_gt = torch.zeros(BATCH_SIZE).long().to(device)
            errD_fake = criterion(prob_matching1, fake_gt)
            D_inputs0 = torch.cat((color_match, labels), dim=1)  # all have their labels
            prob_matching0 = model_D(D_inputs0)  # D try to increase this prob
            real_gt = torch.zeros(BATCH_SIZE).fill_(1).long().to(device)
            errD_real = criterion(prob_matching0, real_gt)

            errD = errD_fake + errD_real
            # errD.backward(retain_graph=True)  # reusing computational graph

            D_fake_loss.update(errD_fake.item(), G_inputs.size(0))
            D_real_loss.update(errD_real.item(), G_inputs.size(0))

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            prob_matching1 = model_D(D_inputs1)  # D try to reduce this prob
            err_G = criterion(prob_matching1, real_gt)
            # err_G.backward(retain_graph=True)

            G_loss.update(err_G.item(), G_inputs.size(0))

            print('validate:[%d/%d] \t  D_fake_loss=%.3f(%.3f) \t D_real_loss=%.3f(%.3f) \t G_loss=%.3f(%.3f)' % (
            batch_i, batch_count,
            D_fake_loss.val, D_fake_loss.avg, D_real_loss.val, D_real_loss.avg, G_loss.val, G_loss.avg))

            D_fake_epoch.append(D_fake_loss.val)
            D_real_epoch.append(D_real_loss.val)
            G_loss_epoch.append(G_loss.val)

            color_diff = cal_diff(G_matching.cpu().detach().numpy(), labels.cpu().detach().numpy())
            diff_ = np.mean(color_diff)
            diff.update(diff_, 64)
            print('diddddddddd')
            print(diff)

    print(diff.avg)
    return D_fake_loss.avg, D_real_loss.avg, G_loss.avg, diff.avg



def cal_diff(concentration, reflectance):
    # the first data -> plot
    # data -> reflectance
    # plot
    Ngrid = BATCH_SIZE
    optical_model = "km"
    sigma = 0.2  # the noise std
    bound = [0., 1., 0., 1.]  # effective bound for likelihood

    color_diff = data.get_lik(concentration, reflectance, n_grid=Ngrid, model=optical_model, sigma=sigma, xvec=None, bound=bound)
    print('++++++++++++++++++++++色差++++++++++++++++++++++++++++++')
    print(color_diff)
    print('----------------------色差------------------------------')
    return color_diff


def save_checkpoint(state, is_best, checkpoint_dir, checkpoint_name, info):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    epoch = state['epoch']
    if is_best:
        best_log_file = os.path.join(checkpoint_dir, "checkpoint.log")
        if not os.path.exists(best_log_file):
            print('checkpoint log file is not exist, creat for the first time')
        with open(best_log_file, 'a+') as f:
            f.writelines(str(epoch) + '\n')
            f.writelines(info + "\n\n\n")
    ckpt_file_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(state, ckpt_file_path)
    print('save checkpoint successfully at %s' % ckpt_file_path)


if __name__ == "__main__":
    main()
