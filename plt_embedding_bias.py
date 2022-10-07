from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot(data_list, pre_pca, alphas, labels, name):
    plt.clf()
    all_samples = np.concatenate(data_list, axis=0)

    if pre_pca:
        all_samples =  PCA(n_components=50).fit_transform(all_samples)
    res = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(all_samples)



    curr_begin= 0
    for i in range(len(data_list)):
        curr_end = curr_begin+data_list[i].shape[0]
        print(curr_begin, curr_end, curr_end-curr_begin)
        plt.plot(res[curr_begin:curr_end, 0], res[curr_begin:curr_end, 1], 'o', alpha=alphas[i], label=labels[i])
        curr_begin = curr_end

    plt.legend()
    plt.savefig(name)

def get_test_train(setting, sel, bias):
    path = "tests/bias_factor_"+str(bias)+"/"+setting+"_selperc"+str(sel)+"_biased/"
    test = np.load(path+'test_queries.npy')
    test_res = np.load(path+'test_res.npy').astype(float).reshape((-1, 1))
    train = np.load(path+'queries.npy')
    res = np.load(path+'res.npy').astype(float).reshape((-1, 1))
    #test = test[:, np.all(~np.isnan(test), axis=0)]
    #train = train[np.all(~np.isnan(train), axis=1)]

    sel_features = False
    if sel_features:
        selector = train
        normalizor = train
        std_thresh = 0.01
        selector = (selector - np.min(selector, axis=0))/(np.max(selector, axis=0)-np.min(selector, axis=0)+1e-5)
        selector_vals = np.std(selector, axis=0)
        test = test[:,np.where(selector_vals > std_thresh)[0]]
        train = train[:,np.where(selector_vals > std_thresh)[0]]
        #normalizor = normalizor[:,np.where(selector_vals > std_thresh)[0]]
        #q_std = np.std(normalizor,axis=0)
        #q_mean = np.mean(normalizor, axis=0)
        #train = (train-q_mean)/q_std
        #test = (test-q_mean)/q_std
    return test, train


#sels = [0.05,0.8]
#biases = [0.6,0.8, 1.0]
settings = ["H1_AVG","H2_AVG","M1_AVG", "M2_AVG"]

for setting in settings:

    test_best, train_best = get_test_train(setting, 0.8, 0.6)

    _, train_10 = get_test_train(setting, 0.05, 1.0)
    _, train_8 = get_test_train(setting, 0.05, 0.8)
    _, train_6 = get_test_train(setting, 0.05, 0.6)

    selector = np.concatenate([train_10, train_8, train_6], axis=0)
    std_thresh = 0.01
    selector = (selector - np.min(selector, axis=0))/(np.max(selector, axis=0)-np.min(selector, axis=0)+1e-5)
    selector_vals = np.std(selector, axis=0)

    test_best = test_best[:,np.where(selector_vals > std_thresh)[0]]

    train_10= train_10[:,np.where(selector_vals > std_thresh)[0]]
    train_8 = train_8[:,np.where(selector_vals > std_thresh)[0]]
    train_6 = train_6[:,np.where(selector_vals > std_thresh)[0]]

    test_true = test_best[0::3]


    plot([train_10, train_8, train_6, test_true], False, [0.1, 0.1, 0.1, 1], ["train_10", "train_8", "train_6", "test_true"], f"{setting}_bias_embed.png")


