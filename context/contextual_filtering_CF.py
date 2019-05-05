import pandas as pd
import numpy as np
import scipy.sparse as sparse
from pyspark import SparkConf, SparkContext
import random
import os, sys, re
sys.path.insert(0, '../item_CF')
import collaborative_filtering as cf

# takes the path of a csv (data_path) and the context of that csv (a description string, context) 
# and write (in recommendations_CF/context) reccomendations for the context for 10 random customers
def get_recommandations(data, context, users_list): 
    #load dataset
    

    users = list(np.sort(data.iloc[:,0].unique()))    # get all unique users
    items = list(np.sort(data.iloc[:,1].unique()))  # get all unique items
    rating = list(np.sort(data.iloc[:,2]))      # get frequencies
    rows = data.iloc[:,0].astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
    cols = data.iloc[:,1].astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

    # Create the purchase matrix as a sparce matrix, each row is a customer and each column is a product
    purchase_matrix = sparse.csc_matrix((rating, (rows, cols)), shape = (len(users), len(items)  )  )


    # transform the matrix into a binary matrix, 1 for purchased and 0 for not purchased. 
    purchase_matrix[purchase_matrix >= 1] =1

    similarity_matrix  = cf.cosine_similarity(purchase_matrix)

    if not os.path.isdir("recommendations_CF/"+context):
            os.mkdir("recommendations_CF/"+context)
            os.mkdir("recommendations_CF/"+context+"/top-10")
            os.mkdir("recommendations_CF/"+context+"/top-100")

    for user in users_list: 
        user_id = user
        user_id_index = users.index(user_id) #used to map real user_id to matrix row
        output = open("recommendations_CF/"+context+"/top-10/top-10_"+str(user_id)+".txt", "w")
        reco = cf.get_N_recommended_items(user_id_index,purchase_matrix,similarity_matrix,items,10)
        output.write("Top-10 recommended products for customer "+str(user_id)+":\nitems_id \t\t similarity\n")
        for rec in reco: 
            output.write(str(rec)+"\n")

        output = open("recommendations_CF/"+context+"/top-100/top-100_"+str(user_id)+".txt", "w")
        output.write("Top-100 recommended products for customer "+str(user_id)+":\nitems_id \t\t similarity\n")
        reco = cf.get_N_recommended_items(user_id_index,purchase_matrix,similarity_matrix,items,100)
        for rec in reco: 
            output.write(str(rec)+"\n")


if not os.path.isdir("recommendations_CF/"):
    os.mkdir("recommendations_CF/")


