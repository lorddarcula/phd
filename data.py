import snscrape.modules.twitter as sntwitter
import pandas as pd
query = "(elon OR musk OR parag OR agarwal OR twitter OR takeover OR fired OR buying OR shutdown) (#elonmusktwitter OR #elonmusktakeover OR #paragagarwal OR #twittershutdown) lang:en until:2022-11-10 since:2022-07-01 -filter:links -filter:replies"
tweets = []
limit = 100000
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])
        tweets.append([tweet.date, tweet.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df)
