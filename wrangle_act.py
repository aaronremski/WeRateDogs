# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # WeRateDogs - Udacity Data Wrangling Project 03
#
#
# ### 8 Quality Issues:
#
# ### 2 Tidiness Issues:

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd
import numpy as np
import os
import requests

import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer



# %% [markdown]
# ## Gather Data #1 - Twitter archive

# %%
twitterDF = pd.read_csv("data/twitter-archive-enhanced.csv")
twitterDF.head(5)

# %%
twitterDF.info()

# %% [markdown]
# ## Gather Data #2 - Tweet image predictions

# %%
file_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
req = requests.get(file_url)
fname = os.path.basename(file_url)
open("data/" + fname, 'wb').write(req.content)

# %% [markdown]
# ## Gather Data #3 - Query Twitter API for additional data
# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
#
#  * retweet count
#  * favorite count
#  * any additional data found that's interesting
#  * only tweets on Aug 1st, 2017 (image predictions present)

# %%

# %%
consumer_key = '3zhhip8Prz2WMil7d0mljLiwI'
consumer_secret = '1aPALPmrd8xBzoovAQabAQJHRCWcijqyu4JJYMzn2S60s6T8iB'
access_token = '175931642-unKWBdyHKcWLgt80FdW1dKFJ3wShN7125I8GsVUy'
access_secret = 'HGRRbxzZKTfsYCiD90jGGyMdKSMuMaLmvWvv74ZDVx2z0'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


# %%
tweet_ids = twitterDF.tweet_id.values
len(tweet_ids)

# %%
# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)

# %% [markdown]
# ### I still do want to understand how to process the JSON 

# %%
with open('tweet.json', "r") as json_file:
    data = json.load(json_file)
    for tweet in data:
        print(f"ID: {tweet['id']}")

# %%
rt_tweets = pd.read_json("tweet.json", lines=True)
rt_tweets.head(5)

# %%
rt_tweets.columns

# %%
rt_tweets[rt_tweets.retweeted == True]

# %%
rt_tweets.iloc[]
