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
# # WeRateDogs Data Wrangling project
# ## 8 Quality Issues:
#
# ## 2 Tidiness Issues:

# %%
import pandas as pd
import numpy as np
import os
import requests

import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer



# %%
twitter = pd.read_csv("data/twitter-archive-enhanced.csv")
twitter.head(5)

# %%
twitter.info()

# %% [markdown]
# ## Tweet image predictions

# %%
file_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
req = requests.get(file_url)
fname = os.path.basename(file_url)
open("data/" + fname, 'wb').write(req.content)

# %% [markdown]
# ## Query Twitter API for additional data
#
#  * retweet count
#  * favorite count
#  * any additional data found that's interesting
#  * only tweets on Aug 1st, 2017 (image predictions present)

# %%

# %%
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

