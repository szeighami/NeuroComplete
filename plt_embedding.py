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
        curr_end = curr_begin+len(data_list[i])
        plt.plot(res[curr_begin:curr_end, 0], res[curr_begin:curr_end:, 1], 'o', alpha=alphas[i], label=labels[i])
        curr_begin = curr_end

    plt.legend()
    plt.savefig(name)




sels = [0.05,0.8]
biases = [0.6,0.8, 1.0]
settings = ["H1_AVG","H2_AVG","M1_AVG", "M2_AVG"]

for setting in settings:
    for bias in biases:
        for sel in sels:
            path = "tests/bias_factor_"+str(bias)+"/"+setting+"_selperc"+str(sel)+"_biased/"
            test = np.load(path+'test_queries.npy')
            test_res = np.load(path+'test_res.npy').astype(float).reshape((-1, 1))
            train = np.load(path+'queries.npy')
            res = np.load(path+'res.npy').astype(float).reshape((-1, 1))
            #test = test[:, np.all(~np.isnan(test), axis=0)]
            train = train[np.all(~np.isnan(train), axis=1)]

            sel_features = True
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


            test_true = test[0::3]
            test_pred = test[2::3]


            plot([train,test_true, test_pred], True, [ 0.1, 0.8, 0.8], ["train", "test_true","test_pred"], f"{setting}_{sel}_{bias}_embed.png")


