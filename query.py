import json
import os
import pandas as pd


class pred_info:
    predicates = None
    predicate_cols = None
    def __init__(self, predicates):
        self.predicates = predicates

data_path = '/home/users/zeighami/neurodb_missing_sigmod/new_try/neurodb_missing/data'

def get_imdb_M2_config(sel_perc, biased_sample_factor):
    config = {}

    setting_path = data_path+'/movies/M2'
    config['selection_percentile'] = sel_perc
    config['biased_sample_factor'] = biased_sample_factor

    config['sel_id_path'] = setting_path+'/biased_sample_factor_' + str(biased_sample_factor) + "/director_sel_perc_"+str(config['selection_percentile'])+".npy"
    config['missing_table_loc'] = setting_path+"/directors.csv"
    config['data_loc'] = setting_path+"/all_tables.csv"

    config['sel_cols'] = ['director_id', 'movie_id', 'company_id', 'actor_id', 'gender', 'birth_year', 'birth_country', 'kind_id', 'genre', 'production_year', 'runtime', 'country', 'company_type_id', 'country_code', 'actor.gender', 'actor.birth_year', 'actor.birth_country']
    config['cat_cols'] = ['gender', 'birth_country', 'kind_id', 'genre', 'country', 'company_type_id', 'country_code', 'actor.gender', 'actor.birth_country']
    config['agg_col'] = 'birth_year'
    config['table_cols_new'] = [['production_year', 'runtime', 'genre', 'country', 'kind_id'], ['company_type_id', 'country_code'], ['actor.gender', 'actor.birth_country', 'actor.birth_year']]
    config['table_cols_id'] = ['movie_id', 'company_id', 'actor_id']
    config['id_col'] = 'director_id'


    config['data_size'] = 103102
    
    return config

def get_imdb_M1_config(sel_perc, biased_sample_factor):
    config = {}

    setting_path = data_path+'/movies/M1'

    config['biased_sample_factor'] = biased_sample_factor
    config['selection_percentile'] = sel_perc

    config['missing_table_loc'] = setting_path+"/movies.csv"
    config['sel_id_path'] = setting_path+'/biased_sample_factor_' + str(biased_sample_factor) + "/movies_sel_perc_"+str(config['selection_percentile'])+".npy"
    config['data_loc'] = setting_path+"/all_tables.csv"

    config['sel_cols'] = ['director_id', 'movie_id', 'company_id', 'actor_id', 'gender', 'birth_year', 'birth_country', 'kind_id', 'genre', 'production_year', 'runtime', 'country', 'company_type_id', 'country_code', 'actor.gender', 'actor.birth_year', 'actor.birth_country']
    config['cat_cols'] = ['gender', 'birth_country', 'kind_id', 'genre', 'country', 'company_type_id', 'country_code', 'actor.gender', 'actor.birth_country']
    config['agg_col'] = 'production_year'
    config['table_cols_new'] = [['gender', 'birth_country', 'birth_year'], ['company_type_id', 'country_code'], ['actor.gender', 'actor.birth_country', 'actor.birth_year']]
    config['table_cols_id'] = ['director_id', 'company_id', 'actor_id']
    config['id_col'] = 'movie_id'

    config['data_size'] = 2528311
    
    return config

def get_airbnb_H1_config(sel_perc, biased_sample_factor):
    config = {}

    setting_path = data_path+'/housing/H1'

    config['selection_percentile'] = sel_perc
    config['biased_sample_factor'] = biased_sample_factor

    config['sel_id_path'] = setting_path+'/biased_sample_factor_' + str(biased_sample_factor) +"/sel_perc_"+str(config['selection_percentile'])+".npy"
    config['missing_table_loc'] = setting_path+"/apartments.csv"
    config['data_loc'] = setting_path+'/all_tables.csv'

    config['sel_cols'] = ['listings.price', 'listings.id', 'hosts.host_response_rate', 'hosts.host_response_time', 'hosts.host_neighbourhood', 'hosts.host_since', 'listings.room_type', 'listings.accommodates', 'listings.property_type', 'neighborhoods.country', 'neighborhoods.neighborhood_id', 'hosts.host_id']
    config['cat_cols'] = ['hosts.host_response_time', 'hosts.host_neighbourhood', 'neighborhoods.country', 'listings.room_type', 'listings.property_type']
    config['agg_col'] = 'listings.price'
    config['table_cols_new'] = [['hosts.host_response_rate', 'hosts.host_response_time', 'hosts.host_since', 'hosts.host_neighbourhood'], ['neighborhoods.country']]
    config['table_cols_id'] = ['hosts.host_id','neighborhoods.neighborhood_id']
    config['id_col'] = 'listings.id'
    
    config['data_size'] = 494925

    return config

def get_airbnb_H2_config(sel_perc, biased_sample_factor):
    config = {}


    setting_path = data_path+'/housing/H2'

    config['selection_percentile'] = sel_perc
    config['biased_sample_factor'] = biased_sample_factor

    config['sel_id_path'] = setting_path+'/biased_sample_factor_' + str(biased_sample_factor) +"/sel_perc_"+str(config['selection_percentile'])+".npy"
    config['missing_table_loc'] = setting_path+"/landlord.csv"
    config['data_loc'] = setting_path+'/all_tables.csv'
    
    config['sel_cols'] = ['listings.price', 'listings.id', 'listings.property_type', 'hosts.host_response_rate', 'hosts.host_response_time', 'hosts.host_neighbourhood', 'hosts.host_since', 'listings.room_type', 'listings.accommodates', 'hosts.host_id', 'neighborhoods.neighborhood_id', 'neighborhoods.country', 'neighborhoods.state']
    config['cat_cols'] = ['hosts.host_response_time', 'hosts.host_neighbourhood', 'neighborhoods.country', 'listings.room_type', 'listings.property_type', 'neighborhoods.state']
    config['agg_col'] = 'hosts.host_response_rate'
    config['table_cols_new'] = [['listings.price', 'listings.room_type', 'listings.property_type', 'listings.accommodates'],['neighborhoods.country']]
    config['table_cols_id'] = ['listings.id', 'neighborhoods.neighborhood_id']
    config['id_col'] = 'hosts.host_id'

    config['data_size'] = 60547

    return config

class raq_base:
    myConfig = None
    exp_name = ""
    def get_default_query_config(self):
        config = None
        with open('default_conf.json') as f:
            config = json.load(f) 

        config['path_to_neurocomplete'] = os.getcwd()

        #We use the same model architecture for both NeuroComplete model and row relevance model, except different epoch numbers
        config['agg_type'] = 1
        config['width'] = 60 # width of the neural network
        config['depth'] = 10 # depth of the neural network
        config['EPOCHS'] = 1000 # number of epochs to train neurocomplete
        config['row_relevance_epochs'] = 100 # number of epochs to train row relevance model
        config['no_batches'] = 100 #number of batches per epoch
        config['lr'] = 0.001
        config['reps'] = 5 #number of reps to train neurocomplete
        config['resample'] = True #number of reps to train neurocomplete


        return config
        

    def set_exp_name(self, query_base_name, sel_perc):
        self.myConfig['exp_name'] = str(query_base_name + "_selperc"+str(sel_perc))+"_embedding_time"
        self.exp_name = str(query_base_name + "_selperc"+str(sel_perc))+"_embedding_time"

        return

    def update_predicate_info(self, predicate_cols, predicates_list):
        self.myConfig['predicates_list'] = predicates_list
        
        return

    def update_query_specific_config(self, update_config):
        self.myConfig.update(update_config)
        return

    def dump_config_file(self):
        exp_dir = 'tests/bias_factor_'+ str(self.myConfig["biased_sample_factor"]) +"/" + self.myConfig["exp_name"]
        os.system('mkdir -p '+exp_dir)

        with open(exp_dir+'/conf.json', 'w') as f:
            json.dump(self.myConfig, f)


    def run_query(self):
        exp_dir = 'tests/bias_factor_'+ str(self.myConfig["biased_sample_factor"]) +"/" + self.myConfig["exp_name"]
        command = 'cd ' + exp_dir + ' && python -u '+ self.myConfig['path_to_neurocomplete'] +'/main.py conf.json  > out.txt  '
        os.system(command)
        command = 'cd ' + self.myConfig['path_to_neurocomplete']
        os.system(command)

        return

    def __init__(self, type):
        self.myConfig = self.get_default_query_config()

def enumerate_H1_COUNT_pred():
    exp_name_base_list = []
    exp_pred_obj_list = []
    exp_base_name = 'H1_COUNT'

    pred_list = [[['listings.price',50,"<"]], [['listings.price',100,"<"]], [['listings.price',150,"<"]], [['listings.price',200,"<"]]]

    pred_info_obj = pred_info(pred_list)
    exp_pred_obj_list.append(pred_info_obj)
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return exp_pred_obj_list, exp_name_base_list

def enumerate_H1_AVG_pred():
    exp_name_base_list = []
    exp_pred_obj_list = []
    exp_base_name = 'H1_AVG'

    pred_list = [[['listings.room_type_1.0',1,"=="]], [['listings.room_type_2.0', 1,"=="]], [['listings.room_type_3.0',1,"=="]], [['listings.room_type_1.0',1,"=="], ['listings.property_type_1.0',1,"=="]], [['listings.room_type_1.0',1,"=="], ['listings.property_type_3.0',1,"=="]], [['listings.room_type_1.0',1,"=="], ['listings.property_type_4.0',1,"=="]], [['listings.room_type_2.0', 1,"=="], ['listings.property_type_1.0',1,"=="]], [['listings.room_type_2.0', 1,"=="], ['listings.property_type_3.0',1,"=="]], [['listings.room_type_2.0', 1,"=="], ['listings.property_type_4.0',1,"=="]], [['listings.room_type_3.0', 1,"=="], ['listings.property_type_1.0',1,"=="]], [['listings.room_type_3.0', 1,"=="], ['listings.property_type_3.0',1,"=="]], [['listings.room_type_3.0', 1,"=="], ['listings.property_type_4.0',1,"=="]], [['listings.property_type_1.0',1,"=="]], [['listings.property_type_3.0',1,"=="]], [['listings.property_type_4.0',1,"=="]],[['listings.accommodates',3,">="]], [['hosts.host_since',2013,">="]]]
    pred_info_obj = pred_info(pred_list)
    exp_pred_obj_list.append(pred_info_obj)
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return exp_pred_obj_list, exp_name_base_list




def enumerate_H2_COUNT_pred():
    pred_list = []
    exp_name_base_list = []
    exp_pred_obj_list = []
    exp_base_name = 'H2_COUNT'
    host_response_time = 2
    host_since = 2011

    predicate_cols = []
    pred_list = [[['hosts.host_response_rate',70,"<"]], [['hosts.host_response_rate',60,"<"]], [['hosts.host_response_rate',80,"<"]], [['hosts.host_response_rate',90,"<"]]]

    pred_info_obj = pred_info(pred_list)
    exp_pred_obj_list.append(pred_info_obj)
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)

    return exp_pred_obj_list, exp_name_base_list

def enumerate_H2_AVG_pred():
    pred_list = []
    exp_name_base_list = []
    exp_base_name = 'H2_AVG'

    predicates = [[['hosts.host_since',2013,">="]], [['hosts.host_response_time_1.0',0,"=="]], [['listings.room_type_1.0',1.0,"=="]],[['listings.room_type_2.0',1.0,"=="]],[['listings.room_type_3.0',1.0,"=="]] ]

    pred_info_obj = pred_info(predicates)
    pred_list.append(pred_info_obj)
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)

    return pred_list, exp_name_base_list


def enumerate_M1_COUNT_pred():
    exp_pred_obj_list = []
    exp_name_base_list = []
    exp_base_name = 'M1_COUNT'

    pred_list = [[['production_year',1960,"<"]], [['production_year',1970,"<"]], [['production_year',1980,"<"]], [['production_year',1990,"<"]]]


    pred_info_obj = pred_info(pred_list)
    exp_pred_obj_list.append(pred_info_obj)
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return exp_pred_obj_list, exp_name_base_list

def enumerate_M1_AVG_pred():
    exp_pred_obj_list = []
    exp_name_base_list = []
    exp_base_name = 'M1_AVG'

    pred_list = [[['genre_Drama',1.0,"=="]], [['birth_country_USA',1.0,"=="]], [['genre_Comedy',1.0,"=="]], [['genre_Documentary',1.0,"=="]], [['genre_Action',1.0,"=="]], [['birth_country_UK',1.0,"=="]], [['birth_country_France',1.0,"=="]], [['birth_country_Germany',1.0,"=="]], [['birth_country_Japan',1.0,"=="]], [['birth_country_Italy',1.0,"=="]], [['birth_country_Canada',1.0,"=="]], [['birth_country_Spain',1.0,"=="]], [['genre_Adventure',1.0,"=="]], [['genre_Animation',1.0,"=="]], [['genre_Crime',1.0,"=="]], [['genre_Horror',1.0,"=="]]]

    pred_info_obj = pred_info(pred_list)
    exp_pred_obj_list.append(pred_info_obj)
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return exp_pred_obj_list, exp_name_base_list


def enumerate_M2_COUNT_pred():
    exp_name_base_list = []
    exp_base_name = 'M2_COUNT'

    predicates = [[['birth_year',1960,"<"]], [['birth_year',1970,"<"]], [['birth_year',1980,"<"]], [['birth_year',1990,"<"]]]
    pred_info_obj = pred_info(predicates)
    pred_list = [pred_info_obj]
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return pred_list, exp_name_base_list


def enumerate_M2_AVG_pred():
    exp_name_base_list = []
    exp_base_name = 'M2_AVG'

    predicates = [[['gender_m',1.0,"=="]]]
    pred_info_obj = pred_info(predicates)
    pred_list = [pred_info_obj]
        
    exp_name = exp_base_name
    exp_name_base_list.append(exp_name)
    
    return pred_list, exp_name_base_list

def get_imdb_M1_queries(sel_percentiles, bias_factors, instances):
    queries = []
    
    if instances:
        for bf, sel_perc in zip(bias_factors, sel_percentiles):
            pred_enum_list, exp_names = enumerate_M1_AVG_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('imdb')
                query.update_query_specific_config(get_imdb_M1_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                queries.append(query)

            pred_enum_list, exp_names = enumerate_M1_COUNT_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('imdb')
                query.update_query_specific_config(get_imdb_M1_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                query.myConfig["agg_type"] = 2
                queries.append(query)
    else:
        for bf in bias_factors:
            for sel_perc in sel_percentiles:
                pred_enum_list, exp_names = enumerate_M1_AVG_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('imdb')
                    query.update_query_specific_config(get_imdb_M1_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    queries.append(query)

                pred_enum_list, exp_names = enumerate_M1_COUNT_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('imdb')
                    query.update_query_specific_config(get_imdb_M1_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    query.myConfig["agg_type"] = 2
                    queries.append(query)

    return queries

def get_imdb_M2_queries(sel_percentiles, bias_factors, instances):
    queries = []

    if instances:
        for bf, sel_perc in zip(bias_factors, sel_percentiles):
            pred_enum_list, exp_names = enumerate_M2_AVG_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('imdb')
                query.update_query_specific_config(get_imdb_M2_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                queries.append(query)

            pred_enum_list, exp_names = enumerate_M2_COUNT_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('imdb')
                query.update_query_specific_config(get_imdb_M2_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                query.myConfig["agg_type"] = 2
                queries.append(query)
    else:
        for bf in bias_factors:
            for sel_perc in sel_percentiles:
                pred_enum_list, exp_names = enumerate_M2_AVG_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('imdb')
                    query.update_query_specific_config(get_imdb_M2_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    queries.append(query)

                pred_enum_list, exp_names = enumerate_M2_COUNT_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('imdb')
                    query.update_query_specific_config(get_imdb_M2_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    query.myConfig["agg_type"] = 2
                    queries.append(query)

    return queries

def get_airbnb_H1_queries(sel_percentiles, bias_factors, instances):
    queries = []
    if instances:
        for bf, sel_perc in zip(bias_factors, sel_percentiles):
            pred_enum_list, exp_names = enumerate_H1_AVG_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('airbnb')
                query.update_query_specific_config(get_airbnb_H1_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                queries.append(query)

            pred_enum_list, exp_names = enumerate_H1_COUNT_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('airbnb')
                query.update_query_specific_config(get_airbnb_H1_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                query.myConfig["agg_type"] = 2
                queries.append(query)
    else:
        for bf in bias_factors:
            for sel_perc in sel_percentiles:
                pred_enum_list, exp_names = enumerate_H1_AVG_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('airbnb')
                    query.update_query_specific_config(get_airbnb_H1_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    queries.append(query)

                pred_enum_list, exp_names = enumerate_H1_COUNT_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('airbnb')
                    query.update_query_specific_config(get_airbnb_H1_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    query.myConfig["agg_type"] = 2
                    queries.append(query)

    return queries

def get_airbnb_H2_queries(sel_percentiles, bias_factors, instances):
    queries = []
    
    if instances:
        for bf, sel_perc in zip(bias_factors, sel_percentiles):
            pred_enum_list, exp_names = enumerate_H2_AVG_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('airbnb')
                query.update_query_specific_config(get_airbnb_H2_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                queries.append(query)

            pred_enum_list, exp_names = enumerate_H2_COUNT_pred()
            for pred_def,exp_name in zip(pred_enum_list, exp_names):
                query = raq_base('airbnb')
                query.update_query_specific_config(get_airbnb_H2_config(sel_perc, bf))
                query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                query.set_exp_name(exp_name, sel_perc)
                query.myConfig["agg_type"] = 2
                queries.append(query)
    else:
        for bf in bias_factors:
            for sel_perc in sel_percentiles:
                pred_enum_list, exp_names = enumerate_H2_AVG_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('airbnb')
                    query.update_query_specific_config(get_airbnb_H2_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    queries.append(query)

                pred_enum_list, exp_names = enumerate_H2_COUNT_pred()
                for pred_def,exp_name in zip(pred_enum_list, exp_names):
                    query = raq_base('airbnb')
                    query.update_query_specific_config(get_airbnb_H2_config(sel_perc, bf))
                    query.update_predicate_info(pred_def.predicate_cols, pred_def.predicates)
                    query.set_exp_name(exp_name, sel_perc)
                    query.myConfig["agg_type"] = 2
                    queries.append(query)


    return queries



class airbnb_query_runner:
    def generate_queries(self, sel_percentiles, bias_factor, queries, instances):
        h1_queries = get_airbnb_H1_queries(sel_percentiles, bias_factor, instances)
        h2_queries = get_airbnb_H2_queries(sel_percentiles, bias_factor, instances)

        all_queries = h1_queries + h2_queries

        if queries is not None:
            applicable_queries = []
        
            for query in all_queries:
                if any(allowed in query.exp_name for allowed in queries):
                    applicable_queries.append(query)
            
            return applicable_queries
        else:
            return all_queries
    
    def run_queries(self, sel_percentiles, bias_factor, queries, instances):
        queries = self.generate_queries(sel_percentiles, bias_factor, queries, instances)
        for query in queries:
            print('Creating config file for query: ' + query.exp_name)
            query.dump_config_file()
        

        
        for query in queries:
            print('Running query: ' + query.exp_name)
            query.run_query()
        

        return
        


class imdb_query_runner:
    def generate_queries(self, sel_percentiles, bias_factor, queries, instances):
        m2_queries = get_imdb_M2_queries(sel_percentiles, bias_factor, instances)
        m1_queries = get_imdb_M1_queries(sel_percentiles, bias_factor, instances)

        all_queries = m2_queries + m1_queries

        if queries is not None:
            applicable_queries = []
        
            for query in all_queries:
                if any(allowed in query.exp_name for allowed in queries):
                    applicable_queries.append(query)
            
            return applicable_queries
        else:
            return all_queries
    
    def run_queries(self, sel_percentiles, bias_factor, queries, instances):
        queries = self.generate_queries(sel_percentiles, bias_factor, queries, instances)
        for query in queries:
            print('Creating config file for query: ' + query.exp_name)
            query.dump_config_file()
        

        for query in queries:
            print('Running query: ' + query.exp_name)
            query.run_query()
        

        return

def main():
    sel_percentiles = [0.05, 0.1, 0.2, 0.4, 0.8]
    bias_factor = [0.6, 0.8, 1.0]
    instances = False

    airbnb_queries = ['H1_COUNT', 'H1_AVG', 'H2_COUNT', 'H2_AVG']
    imdb_queries = ['M1_COUNT', 'M1_AVG', 'M2_COUNT', 'M2_AVG']

    qr = airbnb_query_runner()
    qr.run_queries(sel_percentiles, bias_factor, airbnb_queries, instances)
    qr = imdb_query_runner()
    qr.run_queries(sel_percentiles, bias_factor, imdb_queries, instances)

if __name__=='__main__':
    main()
