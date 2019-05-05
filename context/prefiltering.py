from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import random 
import os
import contextual_filtering

#Run this module to create context csvs (You have to put Customer_product_purchases.csv and transactions.csv in main folder)

#return the csv string rapresentation of a tuple
def fromTuple2CSVLine(data):
  return str(data[0][0]) + "," + str(data[0][1]) + "," + str(data[1]) 

#filter csv with the lamba fun and save it in path/name
def filter_csv(fun, path, name):
    rdd = data.filter(fun)
    rdd = rdd.map(lambda l: ((l[1],l[2]), 1)).reduceByKey(lambda x,y: x+y)
    rdd = rdd.map(fromTuple2CSVLine)
    rdd.coalesce(1).saveAsTextFile(path)
    os.rename(path+"/part-00000", path+"/"+name)


#create  csvs for period day contexts
def create_csv_period_day():

    if not os.path.isdir("data/morning_output") and not os.path.isfile("data/morning_output/morning_customer_product_purchases.csv"):
        filter_csv(lambda l: l[4]==str(0), "data/morning_output", "morning_customer_product_purchases.csv")

    if not os.path.isdir("data/afternoon_output") and not os.path.isfile("data/afternoon_output/afternoon_customer_product_purchases.csv"):
        filter_csv(lambda l: l[4]==str(1), "data/afternoon_output", "afternoon_customer_product_purchases.csv")

    if not os.path.isdir("data/evening_output") and not os.path.isfile("data/evening_output/evening_customer_product_purchases.csv"):
        filter_csv(lambda l: l[4]==str(2), "data/evening_output", "evening_customer_product_purchases.csv")

#create csvs for stations contexts
def create_csv_stations():
    stations = data.map(lambda l: (l[3],1)).reduceByKey(lambda a,b: a+b).collect()
    stations = sorted(stations, key=lambda x: x[1], reverse=True)[:10]

    for station,_ in stations:
        if not os.path.isdir("data/"+station+"_output") and not os.path.isfile("data/"+station+"_output/"+station+"_customer_product_purchases.csv"):
            filter_csv(lambda l: l[3]==station, "data/"+station+"_output", station+"_customer_product_purchases.csv")


if not os.path.isdir("data"):
    os.mkdir("data")

conf = SparkConf().setAppName("lab").setMaster("local[*]")
sc = SparkContext(conf = conf)



data = sc.textFile('../transactions.csv').map(lambda l: l.split(','))


create_csv_period_day()
create_csv_stations()





