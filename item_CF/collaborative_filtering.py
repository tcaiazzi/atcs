import pandas as pd
import numpy as np
import scipy.sparse as sparse
from pyspark import SparkConf, SparkContext
import sklearn.preprocessing as pp
import random
import os


#compute the cosine similarity between the columns of matrix
def cosine_similarity(matrix): 
    col_normed_mat = pp.normalize(matrix.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat

#return all the purchased items of user (user_id)
def get_purchased_items(user_id, purchase_matrix):
    
    return purchase_matrix.getrow(user_id).nonzero()[1]

#return top N similar items to item_id
def get_N_similar_items(item_id, similarity_matrix, N):
    items = similarity_matrix.getrow(item_id)
    items_dic = dict(zip(items.indices, items.data))
    items_dic = sorted(items_dic.items(), key=lambda kv: kv[1], reverse = True)
    return items_dic[:N]

#return top N recommended item for user (user_id)
def get_N_recommended_items(user_id, purchase_matrix, similarity_matrix, items, N): 
    purchased_items = get_purchased_items(user_id, purchase_matrix)
    similar_items = {} 
    for i in purchased_items:      
        for key, value in get_N_similar_items(i, similarity_matrix, N): 
            if key in similar_items:
                newval = value if value > similar_items[key] else similar_items[key]
                similar_items[items[key]] = newval
            else: 
                similar_items[items[key]] = value
    return sorted(similar_items.items(), key=lambda kv: kv[1], reverse = True)[:N]



if __name__ == "__main__":

   

    #load dataset
    data = pd.read_csv('../Customer_product_purchases.csv', sep = ',', encoding= 'iso-8859-1')

    users = list(np.sort(data.CUST_ID.unique()))    # get all unique users
    items = list(np.sort(data.ARTICLE_ID.unique()))  # ger all unique items
    rating = list(np.sort(data.FREQUENCY))      # get frequencies
    rows = data.CUST_ID.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
    cols = data.ARTICLE_ID.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

    # Create the purchase matrix as a sparce matrix, each row is a customer and each column is a product
    purchase_matrix = sparse.csc_matrix((rating, (rows, cols)), shape = (len(users), len(items)  )  )


    # transform the matrix into a binary matrix, 1 for purchased and 0 for not purchased. 
    purchase_matrix[purchase_matrix >= 1] =1

    similarity_matrix  = cosine_similarity(purchase_matrix)


    #recommendation of 10 (an 100) products for 10 random customers  
    for i in range(10): 
        user_id = random.choice(users)
        user_id = users.index(user_id)
        output = open("recommendations/top-10/top-10_"+str(user_id)+".txt", "w")
        reco = get_N_recommended_items(user_id,purchase_matrix,similarity_matrix,items,10)
        output.write("Top-10 recommended products for customer "+str(user_id)+":\nitems_id \t\t similarity\n")
        for rec in reco: 
            output.write(str(rec)+"\n")

        output = open("recommendations/top-100/top-100_"+str(user_id)+".txt", "w")
        output.write("Top-100 recommended products for customer "+str(user_id)+":\nitems_id \t\t similarity\n")
        reco = get_N_recommended_items(user_id,purchase_matrix,similarity_matrix,items,100)
        for rec in reco: 
            output.write(str(rec)+"\n")



