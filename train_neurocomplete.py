import os
import json
import sys
import math
import numpy as np
import pandas as pd

from jax_model import Phi 

import itertools
import time
from functools import partial
import haiku as hk
import jax
from jax import value_and_grad, grad, jit, random, vmap
from jax.experimental import optimizers, stax
import jax.numpy as jnp

f = open('conf.json') 
config = json.load(f) 
base_name = config['exp_name']

test = np.load('test_queries.npy')
test_res = np.load('test_res.npy').astype(float).reshape((-1, 1))
train = np.load('queries.npy')
res = np.load('res.npy').astype(float).reshape((-1, 1))


num_predicates = len(config['predicates_list'])
res_min = res.min()
res_max = res.max()
res_range = res_max-res_min
test_res = (test_res-res_min)/res_range
res = (res-res_min)/res_range

selector = train
normalizor = train
feature_count = 6
std_thresh = 0.01
selector = (selector - np.min(selector, axis=0))/(np.max(selector, axis=0)-np.min(selector, axis=0)+1e-5)
selector_vals = np.std(selector, axis=0)
test = test[:,np.where(selector_vals > std_thresh)[0]]
train = train[:,np.where(selector_vals > std_thresh)[0]]
normalizor = normalizor[:,np.where(selector_vals > std_thresh)[0]]


q_std = np.std(normalizor,axis=0)
q_mean = np.mean(normalizor, axis=0)
train = (train-q_mean)/q_std
test = (test-q_mean)/q_std


model = hk.transform(partial(Phi, out_dim=config['out_dim'], in_dim=config['in_dim'], init_width=config['filter_width1'], mid_width=config['filter_width2'], no_layers=config['phi_no_layers']))

def mse_weighted_loss(model, weights, params, batch):
    inputs, y_true, _weights = batch[0], batch[1], batch[2]
    y_pred = model.apply(params, None, inputs)
    return jnp.average(jnp.square(jnp.subtract(y_pred, y_true)), weights=_weights)


def calc_metrics(model, params, batch, metrics, logs, weights=None):
    x, y_true = batch[0], batch[1]
    y_pred = model.apply(params, None, x)

    for metric in metrics:
        val = metric.calc(y_true, y_pred) # nonunifrom test set
        logs.add(metric.name, val[0])


class Log():
    def __init__(self, save_path="results.json"):
        self.log = {}
        self.save_path = save_path

    def add(self, name, val):
        if name not in self.log:
            self.log[name] = []
        self.log[name].append(float(val))

    def get(self, name):
        return self.log[name][-1]

    def save(self):
        log_df = pd.DataFrame.from_dict(self.log)
        with open(self.save_path, 'w') as f:
            log_df.to_json(f)


class MAE():
    def __init__(self, sel_indx, val_range=1, name="mae"):
        self.name = name
        self.val_range = val_range
        self.sel_indx = sel_indx

    def calc(self, y_true, y_pred):
        y_pred = jnp.clip(y_pred, 0, 1)
        return jnp.average(jnp.abs(y_pred[self.sel_indx::3] - y_true[self.sel_indx::3])*self.val_range, axis=0)


reps = config['reps']
for i in range(reps):
    print("rep", i)
    loss = mse_weighted_loss

    metrics = [MAE(0, res_range, "val_mae_whole_true"), MAE(1, res_range, "val_mae_sample"), MAE(2, res_range, "val_mae_whole_pred")]

    def train_fn(_, i, opt_state, batch):
        params = get_params(opt_state)
        loss_value, grads = value_and_grad(partial(loss, model, weights))(params, batch)
        return opt_update(i, grads, opt_state), loss_value

    key = random.PRNGKey(int(time.time()))
    init_params = model.init(key, train) 
    opt_init, opt_update, get_params = optimizers.adam(config['lr']) 
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    no_batches = config["no_batches"]
    batch_size = train.shape[0]//no_batches


    train_fn = jit(train_fn)
    cum_duration = 0
    logs = Log(base_name+"_"+str(i)+'_hist.json')
    weights = np.ones_like(res)
    print("no batches: ", no_batches)
    for epoch in range(1, config['EPOCHS'] + 1):
        start = time.perf_counter()
        cum_loss = 0
        p = np.random.permutation(len(train))
        train = train[p]
        res = res[p]
        weights = weights[p]
        for batch in range(no_batches):
            mega_batch = (train[batch*batch_size:(batch+1)*batch_size], res[batch*batch_size:(batch+1)*batch_size], weights[batch*batch_size:(batch+1)*batch_size])
            opt_state, loss_value = train_fn(
                key,
                next(itercount),
                opt_state,
                mega_batch
            )

        logs.add("loss", loss_value)
        calc_metrics(model, get_params(opt_state), (test, test_res), metrics, logs)

        duration = time.perf_counter() - start
        cum_duration += duration
        out_str = str(epoch)+" Loss: " + str(loss_value)+" "
        for metric in metrics:
            out_str += metric.name +": " +str(logs.get(metric.name)) +" "


        out_str += " time : " +str(cum_duration) +" "
        print(out_str)

        if epoch % 100 == 0:
            logs.save()

