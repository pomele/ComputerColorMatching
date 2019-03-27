import numpy as np
import torch
import torch.utils.data

ingredients = np.array([
    # [0.3427, 0.4346, 0.4409, 0.4418, 0.4449, 0.4509, 0.4701, 0.5372, 0.6944, 0.8247, 0.8914, 0.9259, 0.9373, 0.9418,
    #  0.9441, 0.9462, 0.9474, 0.948, 0.9483, 0.948, 0.9484, 0.9479, 0.9475, 0.9472, 0.9468, 0.9472, 0.9465, 0.9442,
    #  0.9423, 0.9387, 0.933],
    # [0.2073, 0.2314, 0.2324, 0.2326, 0.2316, 0.2308, 0.23, 0.2295, 0.2283, 0.2268, 0.2266, 0.2254, 0.2244, 0.2231,
    #  0.2221, 0.2208, 0.2202, 0.2193, 0.2183, 0.2171, 0.2163, 0.2147, 0.2133, 0.2119, 0.2107, 0.2094, 0.2086, 0.2077,
    #  0.2035, 0.2023, 0.2005]
    # 09H
    [0.2072717, 0.2314336, 0.2323573, 0.2326192, 0.2315759, 0.2308346, 0.2299640, 0.2295049, 0.2283248, 0.2267959,
     0.2265563, 0.2253986, 0.2243909, 0.2231159, 0.2221052, 0.2208150, 0.2201522, 0.2193270, 0.2182626, 0.2170505,
     0.2162608, 0.2146650, 0.2133304, 0.2119302, 0.2106560, 0.2094081, 0.2085703, 0.2077338, 0.2034682, 0.2023278,
     0.2005492],
    # 08
    [0.3426769, 0.4345741, 0.4408946, 0.4417734, 0.4448756, 0.4509351, 0.4701447, 0.5371643, 0.6943693, 0.8247025,
     0.8914063, 0.9259378, 0.9373208, 0.9417730, 0.9441082, 0.9462348, 0.9473710, 0.9479665, 0.9482779, 0.9480201,
     0.9483707, 0.9478786, 0.9474954, 0.9472089, 0.9468048, 0.9471580, 0.9465347, 0.9441642, 0.9423221, 0.9387001,
     0.9329507],
    # 08S
    [0.2712101, 0.3016053, 0.2955334, 0.2895256, 0.2859294, 0.2884298, 0.2914248, 0.2936781, 0.3083987, 0.3733792,
     0.5332463, 0.7305964, 0.8455071, 0.9001799, 0.9261169, 0.9373263, 0.9413789, 0.9437845, 0.9448064, 0.9448079,
     0.9450820, 0.9446973, 0.9445449, 0.9442747, 0.9437326, 0.9442004, 0.9438083, 0.9414658, 0.9391597, 0.9355036,
     0.9289310],
    # 09B
    [0.3599522, 0.4651332, 0.4698422, 0.4624575, 0.4524772, 0.4450434, 0.4392531, 0.4362574, 0.4357192, 0.4629371,
     0.5635376, 0.7249229, 0.8456681, 0.9069895, 0.9324164, 0.9409078, 0.9437131, 0.9453641, 0.9461564, 0.9460989,
     0.9463242, 0.9454620, 0.9451574, 0.9448677, 0.9443573, 0.9447857, 0.9443947, 0.9421353, 0.9400899, 0.9365174,
     0.9304954],
    # 2704
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472]
])
initial_concentration = np.array([0.51, 0.51, 0.51, 0.51, 0.51])
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

TRAIN_DATA = 200
VALIDATE_DATA = 50
OPTICAL_MODEL = 'km'
BATCH_SIZE = 64
SIGMA = 0.2  # the noise std
YDIM = 31  # number of data samples
BOUND = [0., 1., 0., 1.]  # effective bound for likelihood
SEED = 1  # seed for generating data

def generate(tot_dataset_size, model='km', ydim=31, sigma=0.1, prior_bound=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], seed=0):
    np.random.seed(seed)
    N = tot_dataset_size

    if model == 'km':
        concentrations = np.random.uniform(0, 1, size=(N, 5))
        # concentrations[:, 0] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, 0]
        # concentrations[:, 1] = prior_bound[2] + (prior_bound[3] - prior_bound[2]) * concentrations[:, 1]
        # concentrations[:, 2] = prior_bound[4] + (prior_bound[5] - prior_bound[4]) * concentrations[:, 2]
        # concentrations[:, 3] = prior_bound[6] + (prior_bound[7] - prior_bound[6]) * concentrations[:, 3]
        # concentrations[:, 4] = prior_bound[8] + (prior_bound[9] - prior_bound[8]) * concentrations[:, 4]
        xvec = np.arange(400, 710, (700 - 400) / (ydim - 1))
        xidx = np.arange(0, ydim, 1)
        init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(5, -1)
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.array(
            [concentrations[:, 0] * fst[0, i] + concentrations[:, 1] * fst[1, i] + concentrations[:, 2] * fst[2, i] + concentrations[:, 3] * fst[3, i] + concentrations[:, 4] * fst[4, i] + np.ones(N) * fsb[i]
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


def get_lik(concentration, reflectance, n_grid=BATCH_SIZE, model='km', sigma=None, xvec=None, bound=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
    # mcx = np.linspace(bound[0], bound[1], n_grid)
    # mcy = np.linspace(bound[2], bound[3], n_grid)
    # dmcx = mcx[1] - mcx[0]
    # dmcy = mcy[1] - mcy[0]

    init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(5, -1)
    print('init_conc_arrayinit_conc_arrayinit_conc_arrayinit_conc_arrayinit_conc_arrayinit_conc_arrayinit_conc_array')
    print(init_conc_array)

    diff = np.zeros(n_grid)
    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array

        for i, c in enumerate(concentration):
            fss = c[0] * fst[0] + c[1] * fst[1] + c[2] * fst[2] + c[3] * fst[3] + c[4] * fst[4] + fsb
            # diff[i, :] = np.array([np.sum(((reflectance - (p - ((p + 1) ** 2 - 1) ** 0.5 + 1)) / sigma) ** 2) for p in fss])
            diff[i] = color_diff(reflectance[i], fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1)
        # diff = np.exp(-0.5 * diff)

    elif model == 'four_flux':
        print('Sorry the model have not implemented yet')
        exit(1)

    else:
        print('Sorry no model of that name')
        exit(1)

    # # normalise the posterior
    # diff /= (np.sum(diff.flatten()) * dmcx * dmcy)
    #
    # # compute integrated probability outwards from max point
    # diff = diff.flatten()
    # idx = np.argsort(diff)[::-1]
    # prob = np.zeros(n_grid * n_grid)
    # prob[idx] = np.cumsum(diff[idx]) * dmcx * dmcy
    # prob = prob.reshape(n_grid, n_grid)
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
    for i in range(TRAIN_DATA):
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
            train_data.append(one_data_str)

    print(len(train_data))
    with open('colordata.txt', 'w') as f:
        for line in train_data:
            f.writelines(line + "\n")

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
    generate_validate_file()



