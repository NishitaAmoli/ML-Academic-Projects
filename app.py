from collections import Counter
import plotly.express as px
import plotly.graph_objs as go
from stylecloud import stylecloud
import preprocessor,helper
import sentiment_analysis
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    #converting data to utf-8 format from byte format
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    # st.dataframe(df)

    # fetch unique users, remove 'group_notification and sort users keeping Overall option at the top
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Statistical Analysis
        num_messages, words, num_media_msg, num_links=helper.fetch_stats(selected_user,df)
        st.title("TOP STATISTICS")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_msg)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # daily timeline
        st.title("DAILY TIMELINE")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        plt.plot(daily_timeline['only_date'], daily_timeline['message'], color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # weekly activity map
        st.title("ACTIVITY MAP")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Month")
            busy_month = helper.monthly_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # monthly timeline
        st.title("MONTHLY TIMELINE")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        plt.plot(timeline['time'], timeline['message'],color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # finding busiest users in a group (group level analysis)
        if selected_user== 'Overall':
            st.title('Most Busy Users')
            x, new_df= helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                # ploting bar graph for top 5 active users
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # displaying dataframe having user name and percentage
                st.dataframe(new_df)

        # WordCloud formation for selected user or overall group
        st.title('WordCloud')
        df_wc = helper.create_wordcloud(selected_user,df)
        # forming word cloud of filtered data frame df_wc
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user,df)
        col1, col2 =st.columns(2)

        with col1:
            # displaying data frame having most common words and their frequency
            st.dataframe(most_common_df)
        with col2:
            # drawing horizontal bar graph for most common words and their frequency
            fig, ax = plt.subplots()
            ax.barh(most_common_df['MOST COMMON WORD'], most_common_df['FREQUENCY'])
            plt.xticks(rotation='vertical')

            st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # displaying data frame of emojis occurred in chat and their frequency
            st.dataframe(emoji_df)
        with col2:
            # drawing pie chart for emojis occurred with their percentage value
            fig,ax = plt.subplots()
            ax.pie(emoji_df['FREQUENCY'].head(),labels=emoji_df['EMOJIS'].head(),autopct="%0.2f")
            st.pyplot(fig)

st.sidebar.title("Sentiment Analyzer")
upload_file = st.sidebar.file_uploader("Choose a file", type="txt",key="file_uploader")


if upload_file is not None:
    # converting data to utf-8 format from byte format
    bytes_data = upload_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = sentiment_analysis.preprocess_whatsapp_chat(data)

    #st.dataframe(df)
    df['clean_msg'] = df['message'].apply(sentiment_analysis.clean_text)   # cleaning texts for analysis

    # remove stop words
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()
    df['clean_msg'] = df['clean_msg'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df.groupby('user').agg({'message': 'count',
                                  'emoji': lambda x: ' '.join(set(emoji for emojis in x.dropna() for emoji in emojis))
                                  }).sort_values(by='message', ascending=False)
    #st.dataframe(df)

    sentiments = SentimentIntensityAnalyzer()      # inbuilt function for sentiment analysis

    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["clean_msg"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["clean_msg"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["clean_msg"]]

    st.dataframe(df)

    x = sum(df["positive"])         # calculating sum of +ve, -ve and neutral msg
    y = sum(df["negative"])
    z = sum(df["neutral"])
    # Perform sentiment analysis

    st.title("Polarity of messages:-")
    st.header(sentiment_analysis.score(x, y, z))    # writing output to screen


    stylecloud.gen_stylecloud(' '.join(df['clean_msg']),
                              icon_name='fab fa-whatsapp',
                              colors=['#25D366', '#128C7E', '#075E54'],
                              random_state=13
                             )
    st.title("Analysis of most common words")
    col1, col2 = st.columns(2)
    with col1:
        # calculating most common words by filtering user as group_notification, msg as media omitted and stop words
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()
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
        st.dataframe(most_common_df)
    with col2:
        st.image('stylecloud.png', caption='Chat Cloud',width=500)

    st.title("Emoticons Analysis")
    col1,col2 = st.columns(2)
    fig = go.Figure(data=go.Pie(labels=['Chats without emoji', 'Chats with emoji'],
                                values=
                                df.assign(is_emoji=df['emoji'].apply(lambda x: True if x != '' else False)).groupby(
                                    'is_emoji').count()[['message']].reset_index()['message'],
                                hole=.4, marker=dict(colors=['#25D366', '#075E54', ])))
    fig.update_traces(hoverinfo='label+value')
    with col1:
        st.header("Pie-Chart for emoji analysis")
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        dataframe = pd.DataFrame(Counter([emoji for message in df.emoji for emoji in message]).most_common(),
                     columns=['emoji', 'count'],
                     index=range(1, len(Counter([emoji for message in df.emoji for emoji in message]).most_common()) + 1)
                     )
        st.header("Emoticons with their count")
        st.dataframe(dataframe)

    # Detect media messages
    df['Is_Media'] = df['message'].apply(lambda x: 'Media omitted' in x or '<Media omitted>' in x)

    # Count media messages per sender
    media_counts = df[df['Is_Media']].groupby('user').size().reset_index(name='Media_Count')

    # Create a boxplot
    fig = px.box(media_counts,x='user', y='Media_Count', points='all', title='Media Shared by Users')

    st.title("Media Analysis")
    st.plotly_chart(fig, use_container_width=True)

    st.title("Active hours of users")
    st.subheader("Most active hours of users each day of whole week")
    pivot = pd.pivot_table(df, index='hour', columns='day_name', values='message', aggfunc='count').fillna(0)
    heatmap = go.Heatmap(z=pivot.values,
                         x=pivot.columns,
                         y=pivot.index,
                         hovertemplate='Interventions at %{y}-hour<extra>%{z}</extra>',
                         colorscale='Greens')
    f = go.Figure(data=[heatmap]).update_layout(xaxis={'categoryorder': 'array',
                                                         'categoryarray': ['Monday', 'Tuesday', 'Wednesday',
                                                                           'Thursday', 'Friday', 'Saturday', 'Sunday']})
    st.plotly_chart(f,use_container_width=True)

    # Extract date and time for plotting
    df['only_date'] = df['date'].dt.date
    df['Hour'] = df['date'].dt.hour

    # Group by sender and hour to count messages
    message_flow = df.groupby(['user', 'only_date', 'Hour']).size().reset_index(name='Message_Count')

    # Create a line chart
    fig = px.line(message_flow, x='Hour', y='Message_Count', color='user', line_group='only_date',
                   markers=True)

    # Update layout for better readability
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Messages',
                      xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.title("User Message Responses Flow")
    st.plotly_chart(fig, use_container_width=True)

    # Extract day of the week (0 = Monday, 6 = Sunday)
    df['DayOfWeek'] = df['date'].dt.dayofweek

    # Count messages per day of the week
    message_counts = df.groupby('DayOfWeek').size().reset_index(name='Message_Count')

    # Create a line chart
    fig = px.line(message_counts, x='DayOfWeek', y='Message_Count')

    # Update x-axis to show weekdays
    fig.update_layout(
        xaxis={
            'tickvals': [0, 1, 2, 3, 4, 5, 6],
            'ticktext': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'title': 'Day of the Week'
        },
        yaxis={'title': 'Number of Messages'}
    )
    st.title('User Interventions by Day of the Week')
    st.plotly_chart(fig, use_container_width=True)
















