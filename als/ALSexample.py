from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import random 
import os
conf = SparkConf().setAppName("lab").setMaster("local[*]")
sc = SparkContext(conf = conf)


# Before loading the dataset you will need to map each Customer_id and each Product Id to an integer

if not os.path.isdir("recommendations"):
    os.mkdir("recommendations")
    os.mkdir("recommendations/top-10")
    os.mkdir("recommendations/top-100")


# Load the dataset
data = sc.textFile('../Customer_product_purchases.csv')
header = data.first()
data = data.filter(lambda row: row!=header)
#Map the data into the RDD rating format
ratings = data.map(lambda l: l.split(',')).filter(lambda l: int(l[1]) <= 2147483647).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))) #the filter function is used to avoid java overflow

rank = 10
num_iteration = 12

#Train the model with parameters
model = ALS.trainImplicit(ratings, rank, num_iteration, lambda_=0.01, alpha=0.01)

customers=[]
ratings_list = ratings.collect()
for i in range(0,10):
    customers.append(random.choice(ratings_list)[0])

#recommendation of 10 products for 10 random customers  
for cust in customers: 

    output = open("recommendations/top-10/top-10_"+str(cust)+".txt", "w")
    Reco = model.recommendProducts(cust,10)
    for rec in Reco: 
        output.write(str(rec)+"\n")

    output = open("recommendations/top-100/top-100_"+str(cust)+".txt", "w")
    Reco = model.recommendProducts(cust,100)
    for rec in Reco: 
        output.write(str(rec)+"\n")

    