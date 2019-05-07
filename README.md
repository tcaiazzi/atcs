# ATCS: Recommender Systems - Assignment

# Pre-requisite
To run the scripts you have to put transactions.csv and Customer_product_purchases.csv in root project directory. 

# ALS

The ALS Script uses spark.mllib to train an ALS model on Customer_product_purchase.csv. 
It writes top-10 and top-100 products recommendations (in ALS/recommendations) for 10 random users.

# Item CF

The collaborative_filtering.py script uses pandas, numpy, scipy and sklearn to compute the purchase matrix and the similarity matrix. 
It writes top-10 and top-100 products recommendations (in ALS/recommendations) for 10 random users.

# Context

## prefiltering.py

This script is used to pre-filter the transaction.csv for each context. 
It writes in data/ csvs for period day contexts (morning, afternoon, evening) and for 10 random stations. 

## contextual_filtering_ALS.py

This module contains functions to train ALS models and to get top recommendations. 

## contextual_filtering_CF.py

This module contain the function to get top recommendations using Collaborative Filtering.

## period_day.py 

This script is used to compute top-10 recommendations (for period day contexts) for ten random users using ALS and CF.

N.B. : to run the period_day.py script you'll need to run prefiltering.py 


## stations.py 

This script is used to compute top-10 recommendations (for 10 station contexts) for ten random users using ALS and CF.

N.B. : to run the stations.py script you'll need to run prefiltering.py 


# Result 

