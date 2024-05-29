import preprocessor,helper
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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





