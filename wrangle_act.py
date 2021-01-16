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
# ## 3 datasets
# 1. twitter-archive-enhanced.csv (local archive)
# 2. rt_tweets (obtained data via Twitter API, to get additional fields that coorespond to IDs in twitter_archive)
# 3. image_preds (local archive created from image recognition system
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
# 3. Messy data - variables from both rows and columns --> doggo, floofer, pupper, puppo. Presumably the dog should only have 1 name? If so, this issue can been resolved with imperfection (which name to select when 2 or more given). If not, and multiple 'doggo' names are allowed, then this issue becomes moot.
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
# ## Q3 - doggo, floofer, pupper, & puppo use None; Replace with NaN, or 0, & 1 for present

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
# function to categorize source column

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
twitterDF.sample(2)

# %%
twitterDF.info()

# %%
# Get indices of rows to drop, in this case, any row with a value in retweeted_status_id different that NaN.  
drop_these = twitterDF[twitterDF['retweeted_status_id'].notnull()].index
twitterDF.drop(drop_these,inplace=True)
twitterDF.sample(3)

# %%
# check if any 'notnull' entries exist in retweeted_status_id
twitterDF[twitterDF['retweeted_status_id'].notnull()]

# %%
twitterDF.info()

# %%
# get rid of 3 empty columns representing the retweeted tweets
drop_cols = ['retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp']
twitterDF.drop(drop_cols,axis=1,inplace=True)
twitterDF.info()

# %%
# data exploration
# see sample of is_reply_to_status_id...
twitterDF[twitterDF.in_reply_to_status_id.notnull()]

# %% [markdown]
# ## Gather Data #2 - Tweet image predictions

# %%
# Download data from file_url utilizing requests library & save to line#5
file_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
req = requests.get(file_url)
fname = os.path.basename(file_url)
open("data/" + fname, 'wb').write(req.content)

# %%
# data exploration
# Nows read file downloaded & view sample
image_preds = pd.read_csv("data/image-predictions.tsv", sep="\t")
image_preds.sample(5)

# %%
# data exploration
image_preds.info()

# %% [markdown]
# ## Gather Data #3 - Query Twitter API for additional data
# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
#
#  * retweet count
#  * favorite count
#  * any additional data found that's interesting
#  * only tweets on Aug 1st, 2017 (image predictions present)

# %%
# define keys & API info 
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
# ### Start from here if data already obtained from Twitter

# %%
# Read tweet JSON into dataframe using pandas
# recived ValueError: Trailing data without 'lines=True'

rt_tweets = pd.read_json("tweet.json", lines=True)
rt_tweets.head(5)

# %%
# data exploration
rt_tweets.info()

# %%
# data exploration
rt_tweets[rt_tweets.retweeted_status.notnull()].head(5)

# %%
# data exploration
rt_tweets.user

# %%
# data exploration
rt_tweets.columns

# %% jupyter={"outputs_hidden": true}
# data exploration
# inspect the extended entities data
rt_tweets.loc[0,'extended_entities']

# %% jupyter={"outputs_hidden": true}
# data exploration
# inspect the entities data
rt_tweets.loc[115,'entities']

# %% jupyter={"outputs_hidden": true}
# data exploration
rt_tweets.loc[130,'user']

# %%
# data exploration
rt_tweets.iloc[1:8,11:]

# %% jupyter={"outputs_hidden": true}
# keeping only records of tweets that are NOT retweeted. Should have 2167 after filtering out non-null values of retweeted_status
rt_tweets = rt_tweets[rt_tweets.retweeted_status.isnull()]
rt_tweets.sample(2)

# %%
# add columns to this list for creating a new DF with only columns we want only
tweet_cols = ['created_at','id','full_text','display_text_range','retweet_count','favorite_count','user']

# %%
# create new DF with column defined above
rt_tweets_sub = rt_tweets.loc[:,tweet_cols]
rt_tweets_sub.head(10)

# %%
rt_tweets.drop('retweeted_status',axis=1,inplace=True)
rt_tweets.columns

# %% [markdown]
# ## Merge 3 datasets
#
# 1. twitterDF
# 2. rt_tweets_sub
# 3. image_preds

# %%
# data exploration
twitterDF.info()

# %%
# data exploration
rt_tweets_sub.info()

# %%
image_preds.info()

# %%
# dataframe has a different name for its shared column, id --> tweet_id
rt_tweets_sub = rt_tweets_sub.rename(columns={"id":"tweet_id"})
rt_tweets_sub.head(5)

# %%
# MERGE 2 dataframes!
new_tweets_df = pd.merge(rt_tweets_sub, twitterDF, on='tweet_id')
new_tweets_df.head(3)

# %%
# data exploration
new_tweets_df.info()

# %%
# MERGE newly merged dataframe and image_preds to get new_tweets_df2
new_tweets_df2 = pd.merge(new_tweets_df, image_preds, on='tweet_id')

# %%
# data exploration
new_tweets_df2.head(5)

# %%
# data exploration
new_tweets_df2.info()

# %%
name_by_avgs = new_tweets_df2.groupby("p1")[['p1_conf','rating_numerator','rating_denominator','doggo','floofer','pupper','puppo','favorite_count',
                             'retweet_count']].mean()
name_by_avgs.head(10)

# %%
count_by_name = new_tweets_df2.groupby('p1')['p1_conf'].size()

# %%
count_by_name.head()

# %%
top10_names = count_by_name.sort_values().tail(10)
top10_names

# %%
top10_names.index.values

# %%
top10_val_array = top10_names.values


# %%
top11to20_names = count_by_name.sort_values().tail(20)
top10_names

# %%

# %%
# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
#performance = 3 + 10 * np.random.rand(len(people))

people = top10_names.index.values 

y_pos = np.arange(len(people))

performance = top10_names.values
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Dog Type (predicted) Count ')
ax.set_title('WeRateDogs Dog Breeds represented (top 10)')

plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

names = top10_names.index.values 

'''["225 g flour",
          "90 g sugar",
          "1 egg",
          "60 g butter",
          "100 ml milk",
          "1/2 package of yeast"]
'''

data = top10_names.values

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(names[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("WeRateDogs Top10 Name Distribution")

plt.show()

# %%
name_by_avgs

# %%
new_tweets_df2.iloc[300:305,0:10]

# %%
new_tweets_df2.iloc[300:305,11:20]

 # %%
 '''doggo, floofer, pupper, puppo  '''

desig = ['doggo', 'floofer', 'pupper', 'puppo']

#new_tweets_df2.groupby(desig)[desig].mean()

new_tweets_df2.doggo.mean()

new_tweets_df2[desig].mean()

# %%
new_tweets_df2.name.value_counts()

# %%
