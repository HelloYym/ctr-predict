
# coding: utf-8

# In[1]:


import os
import zipfile
import time
import pickle
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm

from _2_1_gen_user_click_features import add_user_feature_click_hour, add_user_feature_click_day, add_user_feature_click_history, add_user_click_stats
from _2_2_gen_statistics_features import add_feature_click_stats
from _2_3_gen_tricks import add_user_click_rank_day, add_user_click_time_interval_day, add_user_feature_click_rank
from _2_4_gen_predict_category_property import add_property_sim, add_category_predict_rank
from _2_5_ctr_smooth import add_features_smooth_ctr, add_features_day_ctr, add_features_cross_day_ctr, add_features_cross_smooth_ctr
from _2_6_gen_CountVector import add_feature_user_property
from _2_7_gen_click_num import add_feature_click_hour, add_feature_click_day, add_feature_click_history, add_feature_click_day_hour

from utils import load_pickle, dump_pickle, get_feature_value, feature_spearmanr, feature_target_spearmanr, addCrossFeature, calibration
from utils import raw_data_path, feature_data_path, cache_pkl_path, analyse


# In[2]:


def gen_all_data_all_features():

    
    all_data = load_pickle(raw_data_path+'all_data.pkl')
    all_data.drop(['item_category_list', 'item_property_list', 'predict_category_property'], axis=1, inplace=True)
    all_data_path = feature_data_path + 'all_data_all_features.pkl'

#1     #         =======
#    all_data = add_user_feature_click_history(all_data)
    all_data = add_user_feature_click_day(all_data)
    all_data = add_user_feature_click_hour(all_data)
    all_data = add_user_click_stats(all_data)


    
#2     #         =======
    all_data = add_feature_click_stats(all_data)
    
#3     #         ======= 
    all_data = add_user_click_rank_day(all_data)
    all_data = add_user_click_time_interval_day(all_data)
    all_data = add_user_feature_click_rank(all_data)

#4     #         ======= 
  
    all_data = add_property_sim(all_data)
    all_data = add_category_predict_rank(all_data)
    
#5     #         =======
#    all_data = add_features_ctr(all_data, 0)
    all_data = add_features_smooth_ctr(all_data)
    all_data = add_features_day_ctr(all_data)
    all_data = add_features_cross_day_ctr(all_data)
    all_data = add_features_cross_smooth_ctr(all_data)
    
#6     #         =======
    all_data = add_feature_user_property(all_data)

#7     #         =======    
    all_data = add_feature_click_day(all_data)
    all_data = add_feature_click_hour(all_data)
    all_data = add_feature_click_day_hour(all_data)
    
    type_convert = ['item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_page_id', 'shop_id', 'category2_label',
'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level', 'shop_star_level',]
    all_data[type_convert] = all_data[type_convert].astype(np.int64)

    dump_pickle(all_data, all_data_path)

    return all_data


# In[3]:


def final_data_add_features():

    all_data_path = feature_data_path + 'all_data_all_features.pkl'
    all_data = load_pickle(all_data_path)
    
    
    all_data = add_predict_consine_sim(all_data)
    
    type_convert = ['item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_page_id', 'shop_id', 'category2_label', 'item_sales_level', 'item_price_level']

    all_data[type_convert] = all_data[type_convert].astype(np.int64)

    dump_pickle(all_data, all_data_path)

    return all_data


# In[4]:


if __name__ == '__main__':
    all_data = gen_all_data_all_features()
    #all_data = final_data_add_features()
    print(all_data.columns)


# In[ ]:




