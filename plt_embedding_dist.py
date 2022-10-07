from sklearn.manifold import TSNE
import numpy as np
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'legend.frameon': False})
#matplotlib.rcParams.update({'font.sans-serif': ['Helvetica']})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams.update({'text.usetex': False})
matplotlib.rcParams.update({'lines.markersize': 12})

import matplotlib.pyplot as plt

alphabet = {0:'a', 1:'b', 2:'c'}
setting_name = ["H1", "M1"]

def plot(data_list, pre_pca, alphas, labels, title, ax):
    all_samples = np.concatenate(data_list, axis=0)

    if pre_pca:
        all_samples =  PCA(n_components=50).fit_transform(all_samples)
    res = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(all_samples)



    curr_begin= 0
    for i in range(len(data_list)):
        curr_end = curr_begin+data_list[i].shape[0]
        print(curr_begin, curr_end, curr_end-curr_begin)
        if i == len(data_list)-1:
            ax.plot(res[curr_begin:curr_end, 0], res[curr_begin:curr_end, 1], 'o', alpha=alphas[i], label=labels[i], c='k')
        else:
            ax.plot(res[curr_begin:curr_end, 0], res[curr_begin:curr_end, 1], 'o', alpha=alphas[i], label=labels[i])
        curr_begin = curr_end

    ax.set_xlabel("Dim. 1")
    ax.set_ylabel("Dim. 2")
    
    ax.set_title(title)
    #ax.savefig(name)

def get_test_train(setting, sel, bias):
    path = "tests/bias_factor_"+str(bias)+"/"+setting+"_selperc"+str(sel)+"_biased/"
    test = np.load(path+'test_queries.npy')
    test_res = np.load(path+'test_res.npy').astype(float).reshape((-1, 1))
    train = np.load(path+'queries.npy')
    res = np.load(path+'res.npy').astype(float).reshape((-1, 1))

    return test, train

#settings = ["H1_AVG","H2_AVG","M1_AVG", "M2_AVG"]
settings = ["H1_AVG","M1_AVG"]

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16.8, 5.3))

for i, setting in enumerate(settings):

    test_best, train_best = get_test_train(setting, 0.8, 0.6)
    test_worst, train_worst = get_test_train(setting, 0.05, 1.0)

    selector = np.concatenate([train_best, train_worst], axis=0)
    std_thresh = 0.01
    selector = (selector - np.min(selector, axis=0))/(np.max(selector, axis=0)-np.min(selector, axis=0)+1e-5)
    selector_vals = np.std(selector, axis=0)

    test_best = test_best[:,np.where(selector_vals > std_thresh)[0]]
    train_best = train_best[:,np.where(selector_vals > std_thresh)[0]]
    train_worst = train_worst[:,np.where(selector_vals > std_thresh)[0]]

    test_true = test_best[0::3]


    ax = axes[i]
    plot([train_best, train_worst, test_true], False, [0.2, 0.2, 1], ["Train (low bias)", "Train (high bias)", "Test"], f"({alphabet[i]}) {setting_name[i]} embedding", ax)

leg = ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(-0.3, 1.35), fontsize=24, labelspacing=0.1, columnspacing=1, handletextpad=0.1)
for lh in leg.legendHandles: 
    lh.set_alpha(1)


sels = [0.05,0.8]
biases = [0.6,0.8, 1.0]
settings = ["H1_AVG","H2_AVG","M1_AVG", "M2_AVG"]
#settings = ["M2_AVG"]

bar_width = 0.25
low = []
high = []

for setting in settings:

    data_list = []
    for sel in sels:
        for bias in biases:
            path = "tests/bias_factor_"+str(bias)+"/"+setting+"_selperc"+str(sel)+"_biased/"
            train = np.load(path+'queries.npy')
            data_list.append(train)

    test_best, _ = get_test_train(setting, 0.8, 0.6)
    test_true = test_best[0::3]
    data_list.append(test_true)
    
    all_data = np.concatenate(data_list, axis=0)
    selector = all_data
    selector = (selector - np.min(selector, axis=0))/(np.max(selector, axis=0)-np.min(selector, axis=0)+1e-5)
    selector_vals = np.std(selector, axis=0)
    std_thresh =0.01
    mask = np.where(selector_vals > std_thresh)[0]
    all_data = all_data[:, mask]

    test_true = all_data[-len(test_true):]

    i = 0
    curr_begin = 0
    for sel in sels:
        for bias in biases:
            curr_end = curr_begin+len(data_list[i])
            curr_train = all_data[curr_begin:curr_end]
            curr_begin=curr_end
            i += 1

            dist = pairwise_distances(curr_train, test_true)
            sim = cosine_similarity(curr_train, test_true)
            #print(sim)
            max_sim = np.max(sim, axis=0)

            min_dist = np.min(dist, axis=0)
            #print(min_sim.shape)
            #print(min_sim)
            avg_sim =np.mean(max_sim)
            avg_dist =np.mean(min_dist)
            #print(setting, sel, bias, avg_sim, avg_dist)
            if sel == 0.05 and bias == 1.0:
                low.append(avg_dist)
            if sel == 0.8 and bias == 0.6:
                high.append(avg_dist)


br1 = np.arange(len(settings))
br2 = [x + bar_width for x in br1]
ax=axes[2]
ax.bar(br2, high, width = bar_width, label ='Low bias')
ax.bar(br1, low, width = bar_width, label ='High bias')

ax.set_xlabel('Setting')
ax.set_ylabel('Dist. NTS')
ax.set_yscale('log')
ax.set_xticks([r+ bar_width/2 for r in range(len(settings))], ['H1', 'H2', 'M1', "M2"])
ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.45, 1.35), fontsize=24, labelspacing=0.1, columnspacing=1, handletextpad=0.25)
ax.set_title(f"({alphabet[2]}) Embedding distance")
plt.subplots_adjust(left=0.08, bottom=0.17, right=0.98, top=0.82, wspace=0.29)

box = axes[1].get_position()
box.x0 = box.x0 - 0.005
box.x1 = box.x1 - 0.005
axes[1].set_position(box)
box = axes[0].get_position()
box.x0 = box.x0 - 0.007
box.x1 = box.x1 - 0.007
axes[0].set_position(box)

plt.savefig("embeddin_dist.png")
#plt.savefig("embeddin_dist.eps")

