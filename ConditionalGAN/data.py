import numpy as np
import torch
import torch.utils.data
from random import choice

ingredients_1 = np.array([
    # 2704
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    # 09S 黄色
    [0.2877704, 0.3306401, 0.3257476, 0.3195266, 0.3157697, 0.3176763, 0.3205532, 0.3241252, 0.3338507, 0.3885163,
     0.5312571, 0.7235649, 0.8461311, 0.9063247, 0.9318116, 0.9404034, 0.9426319, 0.9439566, 0.9447455, 0.9449095,
     0.9453977, 0.9448299, 0.9444428, 0.9442645, 0.9441012, 0.9446317, 0.9440655, 0.9414658, 0.9391804, 0.9359795,
     0.9293869],
    # 07H
    [0.2072717, 0.2314336, 0.2323573, 0.2326192, 0.2315759, 0.2308346, 0.2299640, 0.2295049, 0.2283248, 0.2267959,
     0.2265563, 0.2253986, 0.2243909, 0.2231159, 0.2221052, 0.2208150, 0.2201522, 0.2193270, 0.2182626, 0.2170505,
     0.2162608, 0.2146650, 0.2133304, 0.2119302, 0.2106560, 0.2094081, 0.2085703, 0.2077338, 0.2034682, 0.2023278,
     0.2005492],
    # 16
    [0.3582383, 0.4589019, 0.4442960, 0.4180548, 0.3943129, 0.3753622, 0.3580040, 0.3467267, 0.3377595, 0.3309189,
     0.3229960, 0.3138701, 0.3117055, 0.3117619, 0.3016199, 0.2815702, 0.2676951, 0.2725082, 0.3257494, 0.5337213,
     0.7479121, 0.8533681, 0.9027772, 0.9256366, 0.9336862, 0.9368781, 0.9375737, 0.9365737, 0.9347985, 0.9322762,
     0.9260198],
    # 08
    # [0.3426769, 0.4345741, 0.4408946, 0.4417734, 0.4448756, 0.4509351, 0.4701447, 0.5371643, 0.6943693, 0.8247025,
    #  0.8914063, 0.9259378, 0.9373208, 0.9417730, 0.9441082, 0.9462348, 0.9473710, 0.9479665, 0.9482779, 0.9480201,
    #  0.9483707, 0.9478786, 0.9474954, 0.9472089, 0.9468048, 0.9471580, 0.9465347, 0.9441642, 0.9423221, 0.9387001,
    #  0.9329507],

    # # 20A-2
    # [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421, 0.7203640, 0.7698454,
    #  0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219, 0.5399272, 0.4776893, 0.4137468, 0.3495456,
    #  0.2899768, 0.2505462, 0.2324735, 0.2232275, 0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188,
    #  0.2783012],

    # 2804
    # [0.3353273, 0.4391780, 0.4634378, 0.4844812, 0.5122489, 0.5352546, 0.5445021, 0.5498481, 0.5548403, 0.5667891,
    #  0.5892876, 0.6205880, 0.6589307, 0.7075461, 0.7588425, 0.8063425, 0.8438276, 0.8656531, 0.8760347, 0.8760190,
    #  0.8742780, 0.8694988, 0.8650322, 0.8608174, 0.8578521, 0.8574630, 0.8571312, 0.8569201, 0.8584669, 0.8579510,
    #  0.8594775]

    #17A
    # [0.4160994, 0.6342332, 0.6881241, 0.6826540, 0.6586606, 0.6262225, 0.5918874, 0.5623005, 0.5190387, 0.4766790,
    #  0.4453364, 0.4108316, 0.3585283, 0.3157409, 0.3090685, 0.3137436, 0.3092155, 0.3295474, 0.4146603, 0.5391598,
    #  0.6621714, 0.7592348, 0.8222232, 0.8629542, 0.8859942, 0.9002361, 0.9062362, 0.9085600, 0.9108428, 0.9079770,
    #  0.9066052]
])
ingredients_2 = np.array([
    # 2704
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    # 09S 黄色
    [0.2877704, 0.3306401, 0.3257476, 0.3195266, 0.3157697, 0.3176763, 0.3205532, 0.3241252, 0.3338507, 0.3885163,
     0.5312571, 0.7235649, 0.8461311, 0.9063247, 0.9318116, 0.9404034, 0.9426319, 0.9439566, 0.9447455, 0.9449095,
     0.9453977, 0.9448299, 0.9444428, 0.9442645, 0.9441012, 0.9446317, 0.9440655, 0.9414658, 0.9391804, 0.9359795,
     0.9293869],
    # 20A-2
    [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421, 0.7203640, 0.7698454,
     0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219, 0.5399272, 0.4776893, 0.4137468, 0.3495456,
     0.2899768, 0.2505462, 0.2324735, 0.2232275, 0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188,
     0.2783012],
    #17A
    [0.4160994, 0.6342332, 0.6881241, 0.6826540, 0.6586606, 0.6262225, 0.5918874, 0.5623005, 0.5190387, 0.4766790,
     0.4453364, 0.4108316, 0.3585283, 0.3157409, 0.3090685, 0.3137436, 0.3092155, 0.3295474, 0.4146603, 0.5391598,
     0.6621714, 0.7592348, 0.8222232, 0.8629542, 0.8859942, 0.9002361, 0.9062362, 0.9085600, 0.9108428, 0.9079770,
     0.9066052]
])
ingredients_3 = np.array([
    # 2704
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    # 08
    [0.3426769, 0.4345741, 0.4408946, 0.4417734, 0.4448756, 0.4509351, 0.4701447, 0.5371643, 0.6943693, 0.8247025,
     0.8914063, 0.9259378, 0.9373208, 0.9417730, 0.9441082, 0.9462348, 0.9473710, 0.9479665, 0.9482779, 0.9480201,
     0.9483707, 0.9478786, 0.9474954, 0.9472089, 0.9468048, 0.9471580, 0.9465347, 0.9441642, 0.9423221, 0.9387001,
     0.9329507],
    # 2804
    [0.3353273, 0.4391780, 0.4634378, 0.4844812, 0.5122489, 0.5352546, 0.5445021, 0.5498481, 0.5548403, 0.5667891,
     0.5892876, 0.6205880, 0.6589307, 0.7075461, 0.7588425, 0.8063425, 0.8438276, 0.8656531, 0.8760347, 0.8760190,
     0.8742780, 0.8694988, 0.8650322, 0.8608174, 0.8578521, 0.8574630, 0.8571312, 0.8569201, 0.8584669, 0.8579510,
     0.8594775],
    #17A
    [0.4160994, 0.6342332, 0.6881241, 0.6826540, 0.6586606, 0.6262225, 0.5918874, 0.5623005, 0.5190387, 0.4766790,
     0.4453364, 0.4108316, 0.3585283, 0.3157409, 0.3090685, 0.3137436, 0.3092155, 0.3295474, 0.4146603, 0.5391598,
     0.6621714, 0.7592348, 0.8222232, 0.8629542, 0.8859942, 0.9002361, 0.9062362, 0.9085600, 0.9108428, 0.9079770,
     0.9066052]
])
initial_concentration = np.array([0.51, 0.51, 0.51, 51])
background = np.array([0.4519, 0.7445, 0.8898, 0.9311, 0.9374, 0.9396, 0.9426, 0.9435, 0.9453, 0.9456, 0.9472, 0.9475,
                       0.9473, 0.9477, 0.9474, 0.9472, 0.9469, 0.9464, 0.9459, 0.945, 0.9446, 0.9434, 0.9428, 0.9422,
                       0.9418, 0.9421, 0.9416, 0.9396, 0.9375, 0.934, 0.9281])
# CIE标准照明体D65光源，10°视场
optical_relevant = np.array([[0.136, 0.667, 1.644, 2.348, 3.463, 3.733, 3.065, 1.934, 0.803, 0.151, 0.036, 0.348, 1.062,
                              2.192, 3.385, 4.744, 6.069, 7.285, 8.361, 8.537, 8.707, 7.946, 6.463, 4.641, 3.109, 1.848,
                              1.053, 0.575, 0.275, 0.120, 0.059],
                             [0.014, 0.069, 0.172, 0.289, 0.560, 0.901, 1.300, 1.831, 2.530, 3.176, 4.337, 5.629, 6.870,
                              8.112, 8.644, 8.881, 8.583, 7.922, 7.163, 5.934, 5.100, 4.071, 3.004, 2.031, 1.295, 0.741,
                              0.416, 0.225, 0.107, 0.046, 0.023],
                             [0.613, 3.066, 7.820, 11.589, 17.755, 20.088, 17.697, 13.025, 7.703, 3.889, 2.056, 1.040,
                              0.548, 0.282, 0.123, 0.036, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                              0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])
perfect_white = np.array([[94.83], [100.00], [107.38]])

# 1000和100
TRAIN_DATA = 6
VALIDATE_DATA = 1
OPTICAL_MODEL = 'km'
BATCH_SIZE = 64
SIGMA = 0.2  # the noise std
YDIM = 31  # number of data samples
BOUND = [0., 1., 0., 1.]  # effective bound for likelihood
SEED = 1  # seed for generating data

noise_1 = 0.1
noise_2 = 0.2
noise_3 = 0.3

def generate(tot_dataset_size, ingredients, model='km', ydim=31, sigma=0.1, prior_bound=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], seed=0):
    np.random.seed(seed)
    N = tot_dataset_size

    if model == 'km':
        concentrations = np.random.uniform(0, 1, size=(N, 4))
        # concentrations[:, 0] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, 0]
        # concentrations[:, 1] = prior_bound[2] + (prior_bound[3] - prior_bound[2]) * concentrations[:, 1]
        # concentrations[:, 2] = prior_bound[4] + (prior_bound[5] - prior_bound[4]) * concentrations[:, 2]
        # concentrations[:, 3] = prior_bound[6] + (prior_bound[7] - prior_bound[6]) * concentrations[:, 3]
        # concentrations[:, 4] = prior_bound[8] + (prior_bound[9] - prior_bound[8]) * concentrations[:, 4]
        # concentrations[:, 5] = prior_bound[10] + (prior_bound[11] - prior_bound[10]) * concentrations[:, 5]
        xvec = np.arange(400, 710, (700 - 400) / (ydim - 1))
        xidx = np.arange(0, ydim, 1)
        init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(4, -1)
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.array(
            [concentrations[:, 0] * fst[0, i] + concentrations[:, 1] * fst[1, i] + concentrations[:, 2] * fst[2, i] +
             concentrations[:, 3] * fst[3, i] +
             # + concentrations[:, 4] * fst[4, i] + concentrations[:, 5] * fst[5, i] +
             np.ones(N) * fsb[i]
             for i in xidx])

        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        reflectance = reflectance.transpose()

    elif model == 'four_flux':
        print('Sorry the model have not implemented yet')
        exit(1)

    else:
        print('Sorry no model of that name')
        exit(1)

    # randomise the data
    shuffling = np.random.permutation(N)
    concentrations = torch.tensor(concentrations[shuffling], dtype=torch.float)
    reflectance = torch.tensor(reflectance[shuffling], dtype=torch.float)

    return concentrations, reflectance, xvec


def get_lik(concentration, reflectance, noise, n_grid=BATCH_SIZE, model='km', sigma=None, xvec=None, bound=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
    if noise == noise_1:
        ingredients = ingredients_1
    elif noise == noise_2:
        ingredients = ingredients_2
    elif noise == noise_3:
        ingredients = ingredients_3

    init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(4, -1)

    # diff = np.zeros(n_grid)
    diff = ''
    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        # for i, c in enumerate(concentration):
        fss = concentration[0] * fst[0] + concentration[1] * fst[1] + concentration[2] * fst[2] + concentration[3] * fst[3] +fsb
        cal_reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        for j in range(len(cal_reflectance)):
            if (cal_reflectance[j] != cal_reflectance[j]):
                cal_reflectance[j] = 0
        diff = color_diff(reflectance, cal_reflectance)

    return diff


def color_diff(reflectance1, reflectance2):
    tri1 = np.dot(optical_relevant, reflectance1.reshape(31, 1))
    tri2 = np.dot(optical_relevant, reflectance2.reshape(31, 1))

    lab1 = xyz2lab(tri1)
    lab2 = xyz2lab(tri2)
    delta_lab = lab1 - lab2

    diff = (delta_lab[0] ** 2 + delta_lab[1] ** 2 + delta_lab[2] ** 2) ** (1 / 2)
    return diff


def xyz2lab(xyz):
    r = 0.008856
    lab = np.zeros(3 * 1)

    if xyz[0] / perfect_white[0] > r and xyz[1] / perfect_white[1] > r and xyz[2] / perfect_white[2] > r:
        lab[0] = (xyz[1] / perfect_white[1]) ** (1 / 3) * 116 - 16
        lab[1] = ((xyz[0] / perfect_white[0]) ** (1 / 3) - (xyz[1] / perfect_white[1]) ** (1 / 3)) * 500
        lab[2] = ((xyz[1] / perfect_white[1]) ** (1 / 3) - (xyz[2] / perfect_white[2]) ** (1 / 3)) * 200
    else:
        lab[0] = (xyz[1] / perfect_white[1]) * 903.3
        lab[1] = (xyz[0] / perfect_white[0] - xyz[1] / perfect_white[1]) * 3893.5
        lab[2] = (xyz[1] / perfect_white[1] - xyz[2] / perfect_white[2]) * 1557.4

    return lab

def print_Tensor_encoded(self, epoch, i, tensors):
    message = '(epoch: %d, iters: %d)' % (epoch, i)
    for k, v in tensors.items():
        with open(self.log_tensor_name, "a") as log_file:
            v_cpu = v.cpu()
            log_file.write('%s: ' % message)
            np.savetxt(log_file, v_cpu.detach().numpy())

def generate_train_file():
    train_data = []
    val_data = []
    for i in range(TRAIN_DATA):
        concentrations_1, reflectance_1, x_1 = generate(
            model=OPTICAL_MODEL,
            tot_dataset_size=BATCH_SIZE,
            ingredients=ingredients_1,
            ydim=YDIM,
            sigma=SIGMA,
            prior_bound=BOUND,
            seed=SEED
        )
        concentrations_2, reflectance_2, x_2 = generate(
            model=OPTICAL_MODEL,
            tot_dataset_size=BATCH_SIZE,
            ingredients=ingredients_2,
            ydim=YDIM,
            sigma=SIGMA,
            prior_bound=BOUND,
            seed=SEED
        )
        concentrations_3, reflectance_3, x_3 = generate(
            model=OPTICAL_MODEL,
            tot_dataset_size=BATCH_SIZE,
            ingredients=ingredients_3,
            ydim=YDIM,
            sigma=SIGMA,
            prior_bound=BOUND,
            seed=SEED
        )

        noisy_1 = torch.zeros(BATCH_SIZE,1).fill_(0.1)
        noisy_2 = torch.zeros(BATCH_SIZE,1).fill_(0.2)
        noisy_3 = torch.zeros(BATCH_SIZE,1).fill_(0.3)

        print(concentrations_1.shape)
        print(noisy_1.shape)

        all_data_1 = torch.cat([reflectance_1, concentrations_1, noisy_1], dim=1)
        all_data_2 = torch.cat([reflectance_2, concentrations_2, noisy_2], dim=1)
        all_data_3 = torch.cat([reflectance_3, concentrations_3, noisy_3], dim=1)

        one_data = np.vstack((all_data_1,all_data_2,all_data_3))

        # concentrations_noise = np.vstack((concentrations_noise_1,concentrations_noise_2,concentrations_noise_3))
        # reflectance = np.vstack((reflectance_1,reflectance_2,reflectance_3))
        #
        # print(concentrations_noise.shape)
        # print(concentrations_noise)
        # # print(concentrations_2)
        #
        # exit(1)
        # one_data = torch.cat([reflectance, concentrations_noise], dim=1)

        for row in one_data:
            one_data_str = ""
            val_data_str = ""
            value = ''
            for val in row:
                value = str(val.item())
                one_data_str += (value + ',')
            one_data_str = one_data_str[:-1]
            print(one_data_str)

            train_data.append(one_data_str)

    with open('colordata.txt', 'w') as f:
        for line in train_data:
            f.writelines(line + "\n")
    with open('val_t_colordata.txt', 'w') as f:
        for i in range(VALIDATE_DATA*BATCH_SIZE):
            line = choice(train_data)
            line = line.split(',')[0:-1]
            line = ','.join(line)
            line_1 =line + ',' + str(noise_1)
            line_2 =line + ',' + str(noise_2)
            line_3 =line + ',' + str(noise_3)
            f.writelines(line_1 + "\n")
            f.writelines(line_2 + "\n")
            f.writelines(line_3 + "\n")


def generate_validate_file():
    validate_data = []
    for i in range(VALIDATE_DATA):
        concentrations, reflectance, x = generate(
            model=OPTICAL_MODEL,
            tot_dataset_size=BATCH_SIZE,
            ydim=YDIM,
            sigma=SIGMA,
            prior_bound=BOUND,
            seed=SEED
        )
        one_data = torch.cat([reflectance, concentrations], dim=1)

        print(one_data.shape)
        for row in one_data:
            one_data_str = ""
            value = ''
            for val in row:
                value = str(val.item())
                one_data_str += (value + ',')
            one_data_str = one_data_str[:-1]
            print(one_data_str)
            validate_data.append(one_data_str)

    print(len(validate_data))
    with open('valcolordata.txt', 'w') as f:
        for line in validate_data:
            f.writelines(line + "\n")


if __name__ == "__main__":

    generate_train_file()
    # generate_validate_file()

