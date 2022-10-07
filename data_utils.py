import numpy as np
import pandas as pd
import json
import operator
import time

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
from utils import mse, MAE, PrintEpochNoBasic
from base_model import Phi

ops = {'==' : operator.eq, ">" : operator.gt, "<" : operator.lt, ">=" : operator.ge,"<=" : operator.le}


def predict_weights(single_table, complete_ids, table_cols, id_col, config):
    weight_col = id_col+'_weights'

    print(single_table.shape)
    sampled_mask = single_table[id_col].isin(complete_ids)

    train_df = single_table[sampled_mask]
    features = train_df[table_cols].to_numpy()
    labels = train_df[weight_col].to_numpy().reshape((-1, 1))
    print(train_df.shape)

    col_mean = np.nanmean(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])


    test_df = single_table[~sampled_mask]
    test_features = test_df[table_cols].to_numpy()
    test_labels = test_df[weight_col].to_numpy().reshape((-1, 1))
    test_index = test_df[id_col].to_numpy().reshape((-1, 1))
    
    col_mean = np.nanmean(test_features, axis=0)
    inds = np.where(np.isnan(test_features))
    test_features[inds] = np.take(col_mean, inds[1])

    f_std = np.std(features, axis=0)
    
    features = features[:, np.where(f_std>0)[0]]
    test_features = test_features[:, np.where(f_std>0)[0]]

    f_mean = np.mean(features, axis=0)
    f_std = np.std(features, axis=0)

    features = (features-f_mean)/f_std
    test_features = (test_features-f_mean)/f_std


    print('Training RR model')
    model = Phi(1, features.shape[1], config['width'], config['width'], config['depth'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    callbacks = [PrintEpochNoBasic(config["print_freq"])]

    loss = mse
    metrics = [MAE(name="error")]

    model.phi.compile(optimizer, loss=loss, metrics=metrics)
    model.phi.fit(features, labels, epochs=10, batch_size=features.shape[0]//config['no_batches'], validation_data=(test_features, test_labels), verbose=0, shuffle=True, callbacks=callbacks)
    pred_weights = model.call(test_features).numpy()

    print("weight pred err", np.mean(np.abs(pred_weights.reshape(-1, 1) - test_labels.reshape(-1, 1))))

    pred_weight_col = weight_col+'_pred'
    pred_df = pd.DataFrame(np.concatenate([test_index.reshape((-1, 1)), pred_weights.reshape((-1, 1))], axis=1), columns=[id_col, pred_weight_col]).set_index(id_col)

    single_table.drop(labels=pred_weight_col, axis=1, inplace=True, errors='ignore')
    single_table = single_table.join(pred_df, id_col)
    single_table.loc[sampled_mask, pred_weight_col] = single_table.loc[sampled_mask, weight_col]
    single_table[pred_weight_col] = single_table[pred_weight_col].fillna(0)

    return single_table


def generate_training_queries(agg_type, db_path,sel_id_path, sel_cols, cat_cols, agg_col,table_cols_new, table_cols_id, id_col, predicates_list, config):
    missing_table_loc = config['missing_table_loc']
    data_size = config['data_size']
    missing_table = pd.read_csv(missing_table_loc, index_col=0)
    orig_df = pd.read_csv(db_path, index_col=0)
    orig_df = orig_df[sel_cols]

    all_qs = []
    all_res = []
    all_test_qs = []
    all_test_res = []

    non_cat_cols = list(set(sel_cols)-set(cat_cols)-set(table_cols_id)-set([id_col]))
    all_new_cat_cols = []
    for col in cat_cols:
        before_cols = orig_df.columns
        orig_df = pd.concat([orig_df,pd.get_dummies(orig_df[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
        after_cols = orig_df.columns
        new_cols = list(set(after_cols)-set(before_cols))
        all_new_cat_cols = all_new_cat_cols+new_cols
        for i in range(len(table_cols_new)):
            if col in table_cols_new[i]:
                table_cols_new[i] = table_cols_new[i] + new_cols
                table_cols_new[i].remove(col)
                break

    orig_df = orig_df[~orig_df[agg_col].isna()]
    orig_df = orig_df.reindex(sorted(orig_df.columns), axis=1)
    missing_table = missing_table[missing_table[id_col].isin(orig_df[id_col].unique())]
    

    startTime = time.time()

    train_preds = []
    for col in all_new_cat_cols:
        train_preds.append([[col, 1, '==']])

    prec_count = 100
    percentiles = [x/prec_count for x in range(1, prec_count)]
    quantiles = orig_df.quantile(q=percentiles, axis=0)
    for col in non_cat_cols:
        quants = quantiles[col].unique().reshape(-1)
        for i in range(quants.shape[0]):
            train_preds.append([[col, quants[i], '>=']])
            train_preds.append([[col, quants[i], '<']])


    endTime = time.time()
    print('generating training predicates took ' + str(endTime-startTime) + ' sec')


    resample = config['resample']

    if resample:
        bias_factor = config["biased_sample_factor"]
        missing_table_size = int(config["selection_percentile"]*missing_table.shape[0])

        biased_sample_size = int(bias_factor*missing_table_size)
        unbiased_sample_size = missing_table_size-int(bias_factor*missing_table_size)

        missing_table = missing_table.sort_values(agg_col)
        all_ids_sorted = missing_table[id_col].values.reshape((-1))
        biased_ids = all_ids_sorted[-biased_sample_size:]

        if bias_factor < 1:
            unbiased_ids = np.random.choice(all_ids_sorted[:-biased_sample_size], size=unbiased_sample_size,replace=False)
            sel_id = np.concatenate([unbiased_ids, biased_ids], axis=0)
        else:
            sel_id = biased_ids
    else:
        sel_id = np.load(sel_id_path)
        missing_table_size = sel_id.shape[0]


    print('Creating Query Embeddings and Answeer')

    startTime = time.time()

    grouped_entity_tables_all = []
    single_tables_all = []
    complete_table_ids = []
    sample_df = orig_df[orig_df[id_col].isin(sel_id)]
    np.save("samples.npy", sel_id)
    for i, table_id_col in enumerate(table_cols_id):
        full_gb = orig_df.groupby([id_col, table_id_col], as_index=False).mean()[table_cols_new[i]+[id_col, table_id_col]]
        grouped_entity_tables_all.append(full_gb)

        full_table = orig_df.groupby([table_id_col], as_index=False).mean()[table_cols_new[i]+[table_id_col]]
        single_tables_all.append(full_table)
        complete_ids = sample_df[table_id_col].unique()
        complete_table_ids.append(complete_ids)
        
    endTime = time.time()
    

    for no_pred in range(len(predicates_list)+len(train_preds)):
        
        if no_pred < len(predicates_list):
            is_test_pred = True
            predicates = predicates_list[no_pred]
        else:
            is_test_pred = False
            predicates = train_preds[no_pred-len(predicates_list)]

        startTime = time.time()
        
        pred_suffix = ""
        df_pred = orig_df
        for pred in predicates:
            predicate_col = pred[0]
            predicate_val = pred[1]
            predicate_cond = pred[2]
            pred_suffix = str(predicate_col) +"_"+ str(predicate_cond)+"_"+ str(predicate_val)
            df_pred = df_pred[ops[predicate_cond](df_pred[predicate_col],predicate_val)]

        pred_ids = df_pred[id_col].unique()
        query_all_df = missing_table[missing_table[id_col].isin(pred_ids)]

        endTime = time.time()
        print("calc ground-truth took", endTime-startTime, "sec")

        train_pred_ids = pred_ids[np.isin(pred_ids, sel_id)]
        query_train_df = missing_table[missing_table[id_col].isin(train_pred_ids)]

        grouped_entity_tables_pred_train = []
        grouped_entity_tables_pred = []
        sample_features= []
        true_features = []
        pred_features = []
        for i, curr_gb in enumerate(grouped_entity_tables_all):
            gb_pred = curr_gb[curr_gb[id_col].isin(pred_ids)]
            gb_pred_train = curr_gb[curr_gb[id_col].isin(train_pred_ids)]

            if agg_type == 1:
                sample_features.append(np.nanmean(gb_pred_train[table_cols_new[i]].values, axis=0).reshape((1, -1)))
            elif agg_type == 2:
                sample_features.append(np.nansum(gb_pred_train[table_cols_new[i]].values, axis=0).reshape((1, -1))/missing_table_size)
            if is_test_pred:
                if agg_type == 1:
                    true_features.append(np.nanmean(gb_pred[table_cols_new[i]].values, axis=0).reshape((1, -1)))
                elif agg_type == 2:
                    true_features.append(np.nansum(gb_pred[table_cols_new[i]].values, axis=0).reshape((1, -1))/data_size)

                grouped_entity_tables_pred.append(gb_pred)
                grouped_entity_tables_pred_train.append(gb_pred_train)

                single_table = single_tables_all[i]

                table_id_col = table_cols_id[i]
                weight_col = table_id_col+'_weights'
                weights = gb_pred[table_id_col].value_counts().rename_axis(table_id_col).to_frame(weight_col)

                single_table.drop(labels=weight_col, axis=1, inplace=True, errors='ignore')
                single_table = single_table.join(weights, on=table_id_col)
                single_table[weight_col] = single_table[weight_col].fillna(0)

                print(complete_table_ids[i].shape)
                single_table = predict_weights(single_table, complete_table_ids[i], table_cols_new[i], table_cols_id[i], config)

                pred_weight_col = table_id_col+'_weights_pred'

                weights = single_table[pred_weight_col].values
                vals = single_table[table_cols_new[i]].values
                ma = np.ma.MaskedArray(vals, mask=np.isnan(vals))
                if agg_type == 1:
                    pred_feature = np.ma.getdata(np.ma.average(ma, weights=weights, axis=0))
                elif agg_type == 2:
                    ma = ma*weights.reshape((-1, 1))
                    pred_feature = np.ma.getdata(np.ma.sum(ma, axis=0))/data_size

                pred_features.append(pred_feature.reshape((1, -1)))

        sample_q = np.concatenate(sample_features, axis=1)
        if is_test_pred:
            all_q = np.concatenate(true_features, axis=1)
            all_q_pred = np.concatenate(pred_features, axis=1)
        else:
            all_q = np.zeros_like(sample_q)
            all_q_pred = np.zeros_like(sample_q)

        endTime = time.time()
        print('embedding and answer predicate '+str(no_pred)+' took ' + str(endTime-startTime) + ' sec')
        startTime = time.time()

        if agg_type == 1:
            sample_res = query_train_df[agg_col].mean()
            test_res = query_all_df[agg_col].mean()
        elif agg_type == 2:
            sample_res = query_train_df.shape[0]/missing_table_size
            test_res = query_all_df.shape[0]/data_size


        all_test_res.append(np.array(test_res).reshape((-1, 1)))
        all_test_res.append(np.array(sample_res).reshape((-1, 1)))
        all_test_res.append(np.array(test_res).reshape((-1, 1)))

        all_test_qs.append(all_q)
        all_test_qs.append(sample_q)
        all_test_qs.append(all_q_pred)
        
    test_queries = np.concatenate(all_test_qs, axis=0)
    test_res = np.concatenate(all_test_res, axis=0)


    queries = test_queries[1::3]
    res = test_res[1::3]
    mask = (res!=0).reshape(-1)
    queries = queries[mask]
    res = res[mask]
    mask = (np.isnan(res)==False).reshape(-1)
    queries = queries[mask]
    res = res[mask]

    test_queries = test_queries[:len(predicates_list)*3]
    test_res = test_res[:len(predicates_list)*3]


    return queries, test_queries, res, test_res



