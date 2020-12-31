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
# ---
# ## 8 Quality Issues 
# Also known as dirty data which includes mislabeled, corrupted, duplicated, inconsistent content issues
#
# ### twitter-archive-enhanced.csv quality issues:
#
# 1. columns 'timestamp' & 'retweeted_status_timestamp' are objects (strings) and not of 'timestamp' type
#
# 2. numerous dog names are "a"; Replace with np.NaN
#    
# 3. doggo, floofer, pupper, & puppo use None; Replace with 0, and 1 where 'doggo, floofer, etc...' 
#
# 4. remove URL from 'source' & replace with 4 categories: iphone, vine, twitter, tweetdeck
#
# 5. remove retweets
#
#
# ---
# ## 2 Tidiness Issues
# Messy data includes structural issues where variables don't form a column, observations form rows, & each observational unit forms a table.
#
#
# ### all 3 datasets tidiness issues:
#
# 1. merge all 3 datasets; remove unwanted columns
#
#
# ### image-predictions.tsv tidiness issues:
#
# 2. Messy data - variables form both rows and columns --> p1, p2, p3, p1_conf, p2_conf, p3_conf, etc. Pivot vars into 3 cols, prediction #, prediction name, prediction probability
#
# 3. Messy data - variables from both rows and columns --> doggo, floofer, pupper, puppo. Presumably the dog should only have 1 name? If so, this issue can been resolved with imperfection (which name to select when 2 or more given). If not, and multiple 'doggo' names are allowed, then is issue becomes moot.
#             

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
# review data columns in DF, are Dtypes appropriate, etc.
twitterDF.info()

# %%
# review names of pups
twitterDF.name.value_counts()

# %%
# review dogtionary names; interesting to see id# 200 has 2 values, doggo & floofer
twitterDF[twitterDF['floofer'] != 'None'].head(3)

# %%
# it appears the designations were pulled from the tweeted text, 'doggo' & 'floofer' in text below
twitterDF.loc[200,'text']

# %%
# Illustrating that pup designations are NOT singular. Multiple 
twitterDF[twitterDF['doggo'] != 'None'].sample(5)

# %% [markdown]
# ## Q1 - Convert dtype of timestamp columns
# Q1 = Quality Item #1

# %%
# Fixed 2 columns with incorrect datatypes, changed to datetime64
twitterDF.timestamp = pd.to_datetime(twitterDF.timestamp)
twitterDF.retweeted_status_timestamp = pd.to_datetime(twitterDF.retweeted_status_timestamp)
twitterDF.info()

# %% [markdown]
# ## Q2 - dog names = 'a', replace with NaN

# %%
# replace puppo's names that match 'a' with NaN
twitterDF.name = np.where(twitterDF.name == 'a', np.NaN, twitterDF.name)

# %%
# check to ensure all 'a' names were removed 
twitterDF[twitterDF.name == 'a']

# %% [markdown]
# ## Q3 - doggo, floofer, pupper, & puppo use None; Replace with NaN, or 0

# %%
# replace 'None' with 0
# replace 'doggo' with 1
twitterDF.doggo = np.where(twitterDF.doggo == 'None', 0, twitterDF.doggo)
twitterDF.doggo = np.where(twitterDF.doggo == 'doggo', 1, twitterDF.doggo)

# %%
# replace 'None' with 0
# replace 'floofer' with 1
twitterDF.floofer = np.where(twitterDF.floofer == 'None', 0, twitterDF.floofer)
twitterDF.floofer = np.where(twitterDF.floofer == 'floofer', 1, twitterDF.floofer)

# %%
# replace 'None' with 0
# replace 'pupper' with 1
twitterDF.pupper = np.where(twitterDF.pupper == 'None', 0, twitterDF.pupper)
twitterDF.pupper = np.where(twitterDF.pupper == 'pupper', 1, twitterDF.pupper)

# %%
# replace 'None' with 0
# replace 'puppo' with 1
twitterDF.puppo = np.where(twitterDF.puppo == 'None', 0, twitterDF.puppo)
twitterDF.puppo = np.where(twitterDF.puppo == 'puppo', 1, twitterDF.puppo)

# %%
# check to ensure cleaning successful
twitterDF[twitterDF.puppo == 'None'].count()

# %%
# check to ensure cleaning successful
twitterDF.query("floofer == 1")

# %% [markdown]
# ## Q4 - remove URL from 'source' & replace with 4 categories: iphone, vine, twitter, tweetdeck

# %%
# review names of sources
twitterDF.source.value_counts()

# %%
twitterDF.head(2)


# %%
def update_source(row):
    if 'iphone' in row:
        return 'iphone'
    elif 'vine' in row:
        return 'vine'
    elif 'Twitter' in row:
        return 'twitter web client'
    elif 'TweetDeck' in row:
        return 'TweetDeck'


# %%
# run update_source function on every row to replace source text with shorter description of source
twitterDF.source = twitterDF.apply(lambda row: update_source(row['source']),axis=1)

# %%
# check to ensure function replaced items as intended
twitterDF.sample(5)

# %% [markdown]
# ## Q5 - remove retweets & delete columns

# %%
# review entries with retweeted ids
twitterDF[pd.notnull(twitterDF.retweeted_status_id)].head(2)

# %%
twitterDF['retweeted_status_id'].notna().count()

# %%
twitterDF = twitterDF[twitterDF['retweeted_status_id'].isnull()]
twitterDF.info()

# %%
drop_cols = ['retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp']
twitterDF.drop(drop_cols,axis=1,inplace=True)
twitterDF.info()

# %%

# %% [markdown]
# ## Gather Data #2 - Tweet image predictions

# %%
file_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
req = requests.get(file_url)
fname = os.path.basename(file_url)
open("data/" + fname, 'wb').write(req.content)

# %%
image_preds = pd.read_csv("data/image-predictions.tsv", sep="\t")
image_preds.sample(5)

# %%
image_preds.info()

# %%

# %% [markdown]
# ## Gather Data #3 - Query Twitter API for additional data
# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
#
#  * retweet count
#  * favorite count
#  * any additional data found that's interesting
#  * only tweets on Aug 1st, 2017 (image predictions present)

# %%
# authenticate API using regenerated keys/tokens

consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

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

# %% jupyter={"outputs_hidden": true}
# remove this code when submitting project
with open('tweet.json', "r") as json_file:
    data = json.load(json_file)
    for tweet in data:
        print(f"ID: {tweet['id']}")

# %%
# Read tweet JSON into dataframe using pandas
# recived ValueError: Trailing data without 'lines=True'

rt_tweets = pd.read_json("tweet.json", lines=True)
rt_tweets.head(5)

# %%
rt_tweets.info()

# %%
rt_tweets[rt_tweets.retweeted_status.notnull()].head(5)

# %%
rt_tweets.user

# %%
rt_tweets.columns

# %%
rt_tweets[rt_tweets.retweeted == True]

# %% jupyter={"outputs_hidden": true}
# inspect the extended entities data
rt_tweets.loc[0,'extended_entities']

# %% jupyter={"outputs_hidden": true}
# inspect the entities data
rt_tweets.loc[115,'entities']

# %% jupyter={"outputs_hidden": true}
rt_tweets.loc[130,'user']

# %% jupyter={"outputs_hidden": true}
rt_tweets.loc[2000,'user']

# %%
rt_tweets.iloc[1:8,0:10]

# %% jupyter={"outputs_hidden": true}
# keeping only records of tweets that are NOT retweeted. Should have 2167 after filtering out non-null values of retweeted_status
rt_tweets = rt_tweets[rt_tweets.retweeted_status.isnull()]
rt_tweets.info()

# %%
rt_tweets.sample(3)

# %%
# add columns to this list for creating a new DF with only column we want only
tweet_cols = ['created_at','id','full_text','display_text_range','retweet_count','favorite_count','user']

# %%
# create new DF with column defined above
tweets_sub = rt_tweets.loc[:,tweet_cols]
tweets_sub.head(10)

# %%
rt_tweets.drop('retweeted_status',axis=1,inplace=True)
rt_tweets.columns

# %%
rt_tweets[rt_tweets.]

# %%

# %%
