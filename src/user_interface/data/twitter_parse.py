'''
@File : twitter_parse.py
@Author : Sudharshana VP
@Contact : sudharshanavp@outlook.com
'''

import tweepy
from src import Constants

# Set Keys and Tokens for Authenticating API access
auth = tweepy.OAuthHandler(Constants.API_CONSUMER_KEY, Constants.API_CONSUMER_SECRET)
auth.set_access_token(Constants.API_ACCESS_TOKEN, Constants.API_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

