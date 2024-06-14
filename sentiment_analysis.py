from datetime import timedelta
import emoji
import pandas as pd
import re

# Function to preprocess the WhatsApp chat log
def preprocess_whatsapp_chat(data):
    # pattern for data having 24 hr time format
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    # splitting message and date from data
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # creating dataframe for msg and date
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type to particular format
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # seperate user name and msg
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['emoji'] = df['message'].apply(lambda x: ''.join(c for c in x if c in emoji.EMOJI_DATA))

    # shift all dates back by one hour (UTC+7)
    df['date'] = df['date'] - timedelta(hours=1)

    # date extraction
    df['hour'] = df['date'].dt.hour
    df['day_name'] = df['date'].dt.day_name()  # emoji extractionme()
    df['week'] = (pd.Timestamp('2024-01-09') - df['date']).dt.days // 7
    return df

def clean_text(text):
    text = text.replace('<Media omitted>', '').replace('This message was deleted', '').replace('\n', ' ').strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[0-9]+','', text)
    text = re.sub(r'\s+',' ', text)
    text = re.sub(r'[^\w\s]|_', '', text)
    text = re.sub(r'([a-zA-Z])\1\1','\\1', text)
    return text.lower()

def score(a,b,c):
    # f'n to calculate sentiments
    outcome=""
    if (a>b) and (a>c):
        outcome="Positive 😊"
    elif (b>a) and (b>c):
        outcome="Negative 😠"
    else:
        outcome="Neutral 🙂"
    return outcome






