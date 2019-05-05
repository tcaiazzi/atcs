import contextual_filtering_CF 
import contextual_filtering_ALS
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from pyspark import SparkConf, SparkContext
import os, re 


def ALS_CF_analysis():

    conf = SparkConf().setAppName("lab").setMaster("local[*]")
    sc = SparkContext(conf = conf)

    data = sc.textFile('../Customer_product_purchases.csv').map(lambda l: l.split(','))
    header = data.first()
    data = data.filter(lambda row: row!=header)

    products_id = data.map(lambda l: l[1]).distinct().collect() #used to map real product id to avoid java overflow

    #write recommendations for each station context in data/ dir
    for directory in os.listdir("data"):
        if re.search('NF.*_', directory):
            station_name = re.split('_', directory)[0]
            station_data = pd.read_csv("data/"+directory+"/"+station_name+"_customer_product_purchases.csv", sep = ',', encoding= 'iso-8859-1')
            station_users = list((station_data.iloc[:,0].unique()))[:10]    # get all unique users
            
            #als analisys 
           
            station_model, station_ratings = contextual_filtering_ALS.create_load_model("data/"+directory+"/"+station_name+"_customer_product_purchases.csv", station_name, sc, products_id)
            for i in range(len(station_users)):
                contextual_filtering_ALS.get_top_N_recommended_products(station_users[i], products_id, station_name, station_model, 10)

            #write recommendations for each station context in data/ dir
            station_name = re.split('_', directory)[0]
            contextual_filtering_CF.get_recommandations(station_data, station_name, station_users)


ALS_CF_analysis()