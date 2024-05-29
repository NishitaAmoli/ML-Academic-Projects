import pandas as pd
from urlextract import  URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
extract=URLExtract()
def fetch_stats(selected_user, df):

      # if selected user is not overall then creating new df for selected user
      if selected_user != 'Overall':
          df= df[df['user'] == selected_user]
      # calculating no. of msg
      num_messages = df.shape[0]
      # calculating words by spliting over space
      words = []
      for message in df['message']:
          words.extend(message.split())
      # calculating number of media messages
      num_media_msg= df[df['message']== '<Media omitted>\n'].shape[0]
      # calculating number of links shared
      links = []
      for message in df['message']:
          links.extend(extract.find_urls(message))
      return num_messages,len(words), num_media_msg, len(links)

# helper f'n to calculate most busy user (bar chart and %)
def most_busy_users(df):
     # calculating user name and their msg count
     x = df['user'].value_counts().head()
     # calculating percentage of user msg
     df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user' : 'percent'})
     return x,df

# helper f'n to create wordcloud
def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    # if user is overall then not creating new df otherwise creating df wrt selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # filtering user as group notification
    temp = df[df['user'] != 'group_notification']
    # filtering message as '<Media omitted>\n'
    temp = temp[temp['message'] != '<Media omitted>\n']

    # f'n to remove stop words stored in file
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    # creating word cloud by filering user as 'overall', msg as 'media omitted' and stop words from file
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='black')
    temp['message'] = temp['message'].apply(remove_stop_words)
    # generating world cloud by splitting on space ' '
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

# helper f'n to calculate most common words
def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    # if user is overall then not creating new df otherwise creating df wrt selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # calculating most common words by filtering user as group_notification, msg as media omitted and stop words
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # creating data frame of most common words by applying filters
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    most_common_df.columns = ['MOST COMMON WORD', 'FREQUENCY']
    return most_common_df

# helper f'n to analysis emojis in chat and their frequency
def emoji_helper(selected_user,df):
    # if user is overall then not creating new df otherwise creating df wrt selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # creating list of emojis in msg
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    # creating data frame of emojis encountered in msg and their frequency
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df.columns = ['EMOJIS', 'FREQUENCY']

    return emoji_df

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def monthly_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_activity_heatmap

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
         df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+ "-"+str(timeline['year'][i]))

    timeline['time'] = time

    return timeline












