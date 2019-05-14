import matplotlib

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_curves(all_data, label):
    data_num = len(all_data)
    D_fake_loss = all_data['D_fake_loss']
    D_real_loss = all_data['D_real_loss']
    G_loss = all_data['G_loss']

    epochs = len(G_loss)

    x = np.array([i for i in range(epochs)])
    D_fake_loss = np.array(D_fake_loss)
    D_real_loss = np.array(D_real_loss)
    G_loss = np.array(G_loss)

    max_loss = max(np.hstack((D_fake_loss, D_real_loss, G_loss)))

    if label == 'train':
        plt.title('training loss curves')
    else:
        plt.title('validate loss curves')

    # plt.plot(x,y)

    plt.plot(x, D_fake_loss, color='cyan', ls='--', label='D_fake_loss')
    plt.plot(x, D_real_loss, 'b', ls='-.', label='D_real_loss')
    plt.plot(x, G_loss, 'g', ls=':', label='G_loss')
    plt.legend()  # show label
    plt.xlabel('epoch')

    if label == 'train':
        plt.ylabel('train loss')

    else:
        plt.ylabel('validate loss')

    plt.axis([0, epochs+10, 0, max_loss])
    if label == 'train':
        plt.savefig('train_loss')
    else:
        plt.savefig('validate_loss')
    plt.show()



def plot_diff(all_data):
    data_num = len(all_data)
    difference = all_data['difference']

    print('epoch_diff')
    print(difference)
    epochs = len(difference)

    x = np.array([i for i in range(epochs)])
    difference = np.array(difference)
    max_diff = max(difference)

    plt.title('validate color diff')
    # plt.plot(x,y)

    plt.plot(x, difference, color='cyan', label='difference')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('diff')
    plt.axis([0, epochs+5, 0, max_diff])
    plt.savefig('diff_curves')
    plt.show()

if __name__ == "__main__":
    plot_curves()
