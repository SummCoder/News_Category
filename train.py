import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from sklearn.svm import LinearSVC
import pickle

# 读取数据
data = pd.read_json('News_Category.json', lines=True)

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

# 特征提取
vectorizer = CountVectorizer(analyzer='word', lowercase=False)

# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# X = tfidf_vectorizer.fit_transform(data['processed_text'])

X = vectorizer.fit_transform(data['processed_text'])
y = data['category']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 训练朴素贝叶斯模型/逻辑回归模型
# nb = MultinomialNB(alpha=0.1)
nb = LogisticRegression(C=1.0, max_iter=500)
# nb = LinearSVC()
nb.fit(X_train, y_train)

# 保存模型
with open('model.pkl', 'wb') as file:
    pickle.dump(nb, file)

# 加载模型
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 保存特征提取器
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)
print('加载的模型准确率:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# print('开始测试')
#
#
# # def predict_category(input_text):
# #     processed_text = preprocess_text(input_text)
# #     processed_text = processed_text + ' ' + extract_link_info(input_text) + ' '  # 这里的input_text是一条记录
# #     X_input = vectorizer.transform([processed_text])
# #     category = nb.predict(X_input)
# #     return category[0]
# #
# #
# # while True:
# #     input_text = input("请输入一条记录（按Enter键输入，输入'exit'退出）: ")
# #     if input_text.lower() == 'exit':
# #         print('程序已退出。')
# #         break
# #     else:
# #         predicted_category = predict_category(input_text)
# #         print('预测的类别:', predicted_category)
#
# # 模型预测
# y_pred = nb.predict(X_test)
#
# # 模型评估
# print('准确率:', accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
