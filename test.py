import pickle
import re

import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_json('测试集.json', lines=True)

# 文本预处理
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z ]+', '', text)  # 去除特殊字符和数字
    words = text.split()
    words = [word for word in words if word not in stop_words]  # 去除停用词
    words = [stemmer.stem(word) for word in words]  # 词干提取
    words = [lemmatizer.lemmatize(word) for word in words]  # 词形还原
    return ' '.join(words)


def extract_link_info(link):
    match = re.search(r'/([^/]+)_', link)
    if match:
        link_info = match.group(1)
        link_info = link_info.replace('-', ' ')  # 将连字符替换为空格
        return link_info
    else:
        return ''


data['link_info'] = data['link'].apply(extract_link_info)
data['processed_text'] = data['short_description']
data['processed_text'] = data['processed_text'].apply(preprocess_text)
data['processed_text'] = data['processed_text'] + ' ' + data['headline']
data['processed_text'] = (data['processed_text'] + ' ' + data['link_info'] + ' ' + data['authors'] + ' ' + data['date'].dt.strftime('%Y-%m-%d'))

# 加载特征提取器
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# X = tfidf_vectorizer.fit_transform(data['processed_text'])

X = vectorizer.transform(data['processed_text'])
y = data['category']

# 加载模型
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X)
print('加载的模型准确率:', accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
# 计算通过的数量
num_correct = (y == y_pred).sum()
print('通过的数量:', num_correct)
