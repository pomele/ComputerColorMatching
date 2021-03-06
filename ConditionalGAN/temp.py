import torch
import torch.nn as nn
import numpy as np

# # ideas_randn = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
# # y = x.repeat(64, 1)
# # noise_index = torch.randint(0, 5, (1,))
# # cur_noise=ideas_randn.index_select(noise_index)
# # noise=cur_noise.repeat(64,1)
#
# ideas_randn = torch.Tensor([0.2, 0.4, 0.6, 0.8, 1.0])
# noise_index = torch.randint(0, 5, (64,))
# cur_noise=torch.index_select(ideas_randn, 0, noise_index)
# cur_noise=cur_noise.unsqueeze(1)
#
# print(cur_noise)
#





# prior_bound=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# concentrations = np.random.uniform(0, 1, size=(64, 6))
# concentrations[:, 0] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, 0]
# concentrations[:, 1] = prior_bound[2] + (prior_bound[3] - prior_bound[2]) * concentrations[:, 1]
# concentrations[:, 2] = prior_bound[4] + (prior_bound[5] - prior_bound[4]) * concentrations[:, 2]
# concentrations[:, 3] = prior_bound[6] + (prior_bound[7] - prior_bound[6]) * concentrations[:, 3]
# concentrations[:, 4] = prior_bound[8] + (prior_bound[9] - prior_bound[8]) * concentrations[:, 4]
# concentrations[:, 5] = prior_bound[10] + (prior_bound[11] - prior_bound[10]) * concentrations[:, 5]
#
# print(concentrations[:, 0])
# print(concentrations)
#
# print(concentrations.shape)

# a = [4, 5]
# b = [3, 9]
# c = [11, 24]
# maxx = max(np.hstack((c, b, a)))
# print(maxx)

# 配色
initial_concentration = np.array([0.51, 0.51, 0.51, 0.51, 0.51, 0.51])
ingredients = np.array([
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
    # 16
    [0.3582383, 0.4589019, 0.4442960, 0.4180548, 0.3943129, 0.3753622, 0.3580040, 0.3467267, 0.3377595, 0.3309189,
     0.3229960, 0.3138701, 0.3117055, 0.3117619, 0.3016199, 0.2815702, 0.2676951, 0.2725082, 0.3257494, 0.5337213,
     0.7479121, 0.8533681, 0.9027772, 0.9256366, 0.9336862, 0.9368781, 0.9375737, 0.9365737, 0.9347985, 0.9322762,
     0.9260198],
    # 20A-2
    [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421, 0.7203640, 0.7698454,
     0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219, 0.5399272, 0.4776893, 0.4137468, 0.3495456,
     0.2899768, 0.2505462, 0.2324735, 0.2232275, 0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188,
     0.2783012],
    # 2704
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    # 2804
    [0.3353273, 0.4391780, 0.4634378, 0.4844812, 0.5122489, 0.5352546, 0.5445021, 0.5498481, 0.5548403, 0.5667891,
     0.5892876, 0.6205880, 0.6589307, 0.7075461, 0.7588425, 0.8063425, 0.8438276, 0.8656531, 0.8760347, 0.8760190,
     0.8742780, 0.8694988, 0.8650322, 0.8608174, 0.8578521, 0.8574630, 0.8571312, 0.8569201, 0.8584669, 0.8579510,
     0.8594775]
])

seed = 0
tot_dataset_size = 64
prior_bound=[0, 1, 0, 1]

np.random.seed(seed)
N = tot_dataset_size
concentrations = np.random.uniform(0, 1, size=(N, 2))
concentrations[:, 0] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, 0]
concentrations[:, 1] = prior_bound[2] + (prior_bound[3] - prior_bound[2]) * concentrations[:, 1]

print(concentrations)

init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(6, -1)
print(init_conc_array)





























