from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
import random 
import os
import pyspark.mllib.recommendation as frusc
import re

# takes an rdd (csv) and returns a ratings rdd
def create_ratings(data, products_id):
    data = data.map(lambda l: Rating(int(l[0]), int(products_id.index(l[1])), float(l[2]))) # products_id.index(l[1]) used to avoid java overflow of Rating fun
    return data

# trains a model on ratings and returns it 
def als_training(ratings, rank=10, num_iteration=12, lambda_=0.01, alpha=0.01):
    model = ALS.trainImplicit(ratings, rank, num_iteration, lambda_=lambda_, alpha=alpha)
    return model 

# write top N products recommendation for the customer (cust_id) and writes it in  recommendations_ALS/context dir
def get_top_N_recommended_products(cust_id, products_id, context, model, N): 
    if not os.path.isdir("recommendations_ALS/"+str(context)):
        os.mkdir("recommendations_ALS/"+str(context))
    output = open("recommendations_ALS/"+str(context)+"/top-10_"+str(cust_id)+".txt", "w")
    output.write("Top-10 recommended products for user "+str(cust_id)+":\n")
    try:
        Reco = model.recommendProducts(cust_id,10)

        for rec in Reco: 
            rec = Rating(rec.user, int(products_id[rec.product]), rec.rating)  #int(products_id[rec.product]) used to restore real product_id
            output.write(str(rec)+"\n")
    except: 
         output.write("The user "+str(cust_id)+" has not bought any product in the context "+ context+"\n")

# creates (if doesn't exists) or loads model (model_name) for the context of csv (data_path)
def create_load_model(data_path, model_name, sc, products_id): 
    data =  sc.textFile(data_path).map(lambda l: l.split(','))
    ratings = create_ratings(data, products_id)

    if not os.path.isdir("models/"+model_name): 
        print("\n\nsaving model ...\n\n")
        model = als_training(ratings)
        model.save(sc, "models/"+model_name)
    else:
        print("\n\nloading model ...\n\n")
        model = MatrixFactorizationModel.load(sc, "models/"+model_name)

    return model,ratings


if not os.path.isdir("recommendations_ALS/"):
        os.mkdir("recommendations_ALS/")

if not os.path.isdir("models/"):
        os.mkdir("models/")




