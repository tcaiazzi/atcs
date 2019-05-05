import contextual_filtering_CF 
import contextual_filtering_ALS
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from pyspark import SparkConf, SparkContext


evening_data = pd.read_csv('data/evening_output/evening_customer_product_purchases.csv', sep = ',', encoding= 'iso-8859-1')
evening_users = list((evening_data.iloc[:,0].unique()))[:10]    # get all unique users
afternoon_data = pd.read_csv('data/afternoon_output/afternoon_customer_product_purchases.csv', sep = ',', encoding= 'iso-8859-1')
afternoon_users = list((afternoon_data.iloc[:,0].unique()))[:10]     # get all unique users
morning_data = pd.read_csv('data/morning_output/morning_customer_product_purchases.csv', sep = ',', encoding= 'iso-8859-1')
morning_users = list((morning_data.iloc[:,0].unique()))[:10]     # get all unique users

def CF_analysis():
    contextual_filtering_CF.get_recommandations(evening_data, "evening", evening_users)
    contextual_filtering_CF.get_recommandations(afternoon_data, "afternoon", afternoon_users)
    contextual_filtering_CF.get_recommandations(morning_data, "morning", morning_users)

def ALS_analysis():

    conf = SparkConf().setAppName("lab").setMaster("local[*]")
    sc = SparkContext(conf = conf)

    data = sc.textFile('../Customer_product_purchases.csv').map(lambda l: l.split(','))
    header = data.first()
    data = data.filter(lambda row: row!=header)

    products_id = data.map(lambda l: l[1]).distinct().collect() #used to map real product id to avoid java overflow

    evening_model, evening_ratings = contextual_filtering_ALS.create_load_model('data/evening_output/evening_customer_product_purchases.csv', "evening_model",sc, products_id)
    afternoon_model, afternoon_ratings  = contextual_filtering_ALS.create_load_model('data/afternoon_output/afternoon_customer_product_purchases.csv', "afternoon_model",sc, products_id)
    morning_model, morning_ratings = contextual_filtering_ALS.create_load_model('data/morning_output/morning_customer_product_purchases.csv', "morning_model",sc, products_id)

    for i in range(len(morning_users)): 
        contextual_filtering_ALS.get_top_N_recommended_products(evening_users[i], products_id, "evening", evening_model, 10)
        contextual_filtering_ALS.get_top_N_recommended_products(afternoon_users[i], products_id, "afternoon", afternoon_model, 10)
        contextual_filtering_ALS.get_top_N_recommended_products(morning_users[i], products_id, "morning", morning_model, 10)

ALS_analysis()
CF_analysis()