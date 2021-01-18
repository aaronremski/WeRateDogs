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
# ## Gather 3 datasets from 3 different sources:
# 1. [Gather Data #1](#gather1) - Twitter archive, twitter-archive-enhanced.csv (local archive)
# 2. [Gather Data #2](#gather2) - Tweet image predictions - rt_tweets (obtained data via Twitter API, to get additional fields that coorespond to IDs in twitter_archive)
# 3. [Gather Data #3](#gather3) - Query Twitter API for additional data - image_preds (local archive created from image recognition system)
#
# ## 8 Quality Issues 
# Also known as dirty data which includes mislabeled, corrupted, duplicated, inconsistent content issues
#
# ### twitter-archive-enhanced.csv quality issues:
#
# 1. [Quality #1](#q1) - columns 'timestamp' & 'retweeted_status_timestamp' are objects (strings) and not of 'timestamp' type
#
# 2. [Quality #2](#q2) - numerous dog names are "a"; Replace with np.NaN
#    
# 3. [Quality #3](#q3) - doggo, floofer, pupper, & puppo use None; Replace with 0, and 1 where 'doggo, floofer, etc...' 
#
# 4. [Quality #4](#q4) - remove URL from 'source' & replace with 4 categories: iphone, vine, twitter, tweetdeck
#
# 5. [Quality #5](#q5) - remove retweets
#
#
# ---
# ## 2 Tidiness Issues
# Messy data includes structural issues where variables don't form a column, observations form rows, & each observational unit forms a table.
#
#
# ### all 3 datasets tidiness issues:
#
# 1. [Tidy #1](#t1) - merge all 3 datasets; remove unwanted columns
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
# %matplotlib inline


# %% [markdown]
# ## <a name="gather1">Gather Data #1 - Twitter archive</a>

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
# ## <a name="q1"> Q1 - Convert dtype of timestamp columns</a>
#
# Q1 = Quality Item #1

# %%
# Fixed 2 columns with incorrect datatypes, changed to datetime64
twitterDF.timestamp = pd.to_datetime(twitterDF.timestamp)
twitterDF.retweeted_status_timestamp = pd.to_datetime(twitterDF.retweeted_status_timestamp)
twitterDF.info()

# %% [markdown]
# ## <a name="q2"> Q2 - dog names = 'a', replace with NaN </a>

# %%
# replace puppo's names that match 'a' with NaN
twitterDF.name = np.where(twitterDF.name == 'a', np.NaN, twitterDF.name)

# %%
# check to ensure all 'a' names were removed 
twitterDF[twitterDF.name == 'a']

# %% [markdown]
# ## <a name="q3"> Q3 - doggo, floofer, pupper, & puppo use None; Replace with NaN, or 0, & 1 for present </a>

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
# ## <a name="q4"> Q4 - remove URL from 'source' & replace with 4 categories: iphone, vine, twitter, tweetdeck </a>

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
# ## <a name="q5"> Q5 - remove retweets & delete columns </a>

# %%
twitterDF.sample(2)

# %%
# Get indices of rows to drop, in this case, any row with a value in retweeted_status_id different that NaN.  
drop_these = twitterDF[twitterDF['retweeted_status_id'].notnull()].index
twitterDF.drop(drop_these,inplace=True)
twitterDF.sample(3)

# %%
# check if any 'notnull' entries exist in retweeted_status_id
twitterDF[twitterDF['retweeted_status_id'].notnull()]

# %%
# get rid of 3 empty columns representing the retweeted tweets
drop_cols = ['retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp']
twitterDF.drop(drop_cols,axis=1,inplace=True)
twitterDF.info()

# %%
# check to ensure cols dropped
twitterDF.info()

# %%
# data exploration
# see sample of is_reply_to_status_id...
twitterDF[twitterDF.in_reply_to_status_id.notnull()]

# %% [markdown]
# ## <a name="gather2">Gather Data #2 - Tweet image predictions</a>

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
# ## <a name="gather3">Gather Data #3 - Query Twitter API for additional data</a>
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
# ## <a name="#t1">Merge 3 datasets</a>
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
new_tweets_df2.name.isnull().count()

# %%
# data exploration
new_tweets_df2.loc[576,'expanded_urls']

# %%
# data exploration
new_tweets_df2.info()

# %%
count_by_name = new_tweets_df2.groupby('p1')['p1_conf'].size()

# %%
count_by_name.sort_values(ascending=False)

# %%
top10_names = count_by_name.sort_values(ascending=False).head(10)
top10_names

# %%
top10_names.index.values

# %%
top10_val_array = top10_names.values


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
# Data Exploration
new_tweets_df2.iloc[300:305,0:10]

# %%
# Data Exploration
new_tweets_df2.iloc[300:305,11:20]

# %%
## Data Exploration 
## Averages of doggo, floofer, pupper, & puppo. Essentially, how often have these been designated

## This means that 'doggo' was used to describe a pup 3.6% of the time

desig = ['doggo', 'floofer', 'pupper', 'puppo']

#new_tweets_df2.groupby(desig)[desig].mean()

new_tweets_df2.doggo.mean()

new_tweets_df2[desig].mean()

# %%
## Data Exploration 
## Names most used

new_tweets_df2.name.value_counts()

# %%
top10_names.index.values

# %%
# Create grouping of dataframe on the first predicted name, p1, & obtained the mean of specific data points

# This one provides appropriate columns but it correctly displayed the resulting dataframe in p1 alphabetic order
# which is not statistically significant
#name_by_avgs = new_tweets_df2.groupby("p1")[['p1_conf','rating_numerator','rating_denominator','doggo','floofer','pupper','puppo','favorite_count',
#                             'retweet_count']].mean()

name_by_avgs = new_tweets_df2.groupby(top10_names.index.values)[['p1_conf','rating_numerator','rating_denominator','doggo','floofer','pupper','puppo','favorite_count',
                             'retweet_count']].mean()


name_by_avgs.head(10)

# %%
name_by_avgs.index

# %%
name_by_avgs

# %%
#name_by_avgs.reset_index(inplace=True)

# %%
#name_by_avgs.rename(columns= {'p1':'probable_name', 'p1_conf':'probability'}, inplace=True)

# %%
favorites_by_name = name_by_avgs.loc[:,['favorite_count']]
favorites_by_name

# %%
top15_favorites = favorites_by_name.iloc[0:15,:]
top15_favorites

# %% [markdown]
# ## Notable analysis from visual bar chart
#
# ### None of the top 15 favorited 'dogs' were acturately identified as dogsl

# %%
# create sub
favorites_by_name = name_by_avgs.loc[:,['favorite_count']]
favorites_by_name.sort_values(by=['favorite_count'], ascending=False, inplace=True)
# get top 15 of new subset to create visual from
top15_favorites = favorites_by_name.iloc[0:15,:]
group_names = top15_favorites.index
group_data = top15_favorites.favorite_count

# %%
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-10000, 70000], xlabel='No. of favorited tweets', ylabel='Names (guessed by learning model)',
       title='Top 15 Favorites (tweets), by probable name')

plt.show;

# %%
name_by_avgs

# %%
# obtained directly from matplotlib examples @ 
# https://matplotlib.org/gallery/pie_and_polar_charts/bar_of_pie.html#sphx-glr-gallery-pie-and-polar-charts-bar-of-pie-py

## Attempt at Bar of Pie visualization

# make figure and assign axis objects
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(wspace=0)

'''
# pie chart parameters
ratios = [.27, .56, .17]
labels = ['Approve', 'Disapprove', 'Undecided']
explode = [0.1, 0, 0]
# rotate so that first wedge is split by the x-axis
angle = -180 * ratios[0]
ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
        labels=labels, explode=explode)
'''

# pie chart parameters


# rotate so that first wedge is split by the x-axis
angle = -180 * ratios[0]
ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
        labels=labels, explode=explode)


# bar chart parameters

xpos = 0
bottom = 0
ratios = [.33, .54, .07, .06]
width = .2
colors = [[.1, .3, .5], [.1, .3, .3], [.1, .3, .7], [.1, .3, .9]]

for j in range(len(ratios)):
    height = ratios[j]
    ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
    ypos = bottom + ax2.patches[j].get_height() / 2
    bottom += height
    ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
             ha='center')

ax2.set_title('Age of approvers')
ax2.legend(('50-65', 'Over 65', '35-49', 'Under 35'))
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
# get the wedge data
theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
center, r = ax1.patches[0].center, ax1.patches[0].r
bar_height = sum([item.get_height() for item in ax2.patches])

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(4)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(4)

plt.show()
