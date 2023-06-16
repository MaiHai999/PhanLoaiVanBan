

from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set()
import joblib
from sklearn.svm import LinearSVC


#đọc file và lấy dữ liệu
path_w = "../pre_data/pre_contruction.txt"
with open(path_w,encoding="UTF-8") as f:
    raw_contruction = f.read()

path_w = "../pre_data/pre_economics.txt"
with open(path_w,encoding="UTF-8") as f:
     raw_economics = f.read()

path_w = "../pre_data/pre_food.txt"
with open(path_w,encoding="UTF-8") as f:
     raw_food = f.read()

path_w = "../pre_data/pre_history.txt"
with open(path_w,encoding="UTF-8") as f:
     raw_history = f.read()

path_w = "../pre_data/pre_medican.txt"
with open(path_w,encoding="UTF-8") as f:
     raw_medican = f.read()


#phân tích thành từng câu và tạo nhãn
raw_contruction = nltk.sent_tokenize(raw_contruction)
lable_contruction = []
for i in range(len(raw_contruction)):
    lable_contruction.append("contruction")

raw_economics = nltk.sent_tokenize(raw_economics)
lable_economics = []
for i in range(len(raw_economics)):
    lable_economics.append("economics")


raw_food = nltk.sent_tokenize(raw_food)
lable_food = []
for i in range(len(raw_food)):
    lable_economics.append("food")



raw_history = nltk.sent_tokenize(raw_history)
lable_history = []
for i in range(len(raw_history)):
    lable_history.append("history")



raw_medican = nltk.sent_tokenize(raw_medican)
lable_medican = []
for i in range(len(raw_medican)):
    lable_medican.append("medician")



#tạo tập train và tập test
x = raw_medican + raw_food + raw_history + raw_contruction + raw_economics
y = lable_medican + lable_food + lable_history + lable_contruction + lable_economics

#phân chia tập test và tập train
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

#khôi phục model embedding
filename = 'C:\\Users\\HP\\PycharmProjects\\embeding_text_classcifition\\vectors.kv'
model1 = joblib.load(filename)


def embedding_feats(list_of_lists):
    DIMENSION =300
    feats = []
    for tokens in list_of_lists:
        tokens1 = nltk.word_tokenize(tokens)
        feat_for_this = np.zeros(DIMENSION)
        count_for_this = 0
        for token in tokens1:
            if token in model1.wv:
                feat_for_this += model1.wv[token]
                count_for_this += 1
        sum = feat_for_this / count_for_this
        feats.append(sum)

    return feats


#biến đổi văn bản thành vecto
X_train_vectors = embedding_feats(X_train)
X_test_vectors = embedding_feats(X_test)


#khỏi tạo model phân loại
sv = LinearSVC(class_weight='balanced')
nb = sv.fit(X_train_vectors, y_train)#train the mode


#dự đoán
y_pred_class = nb.predict(X_test_vectors)
print(y_pred_class)


#vẽ sơ đồ
b = ['contruction','economics','food','history','medician']
mat = confusion_matrix(y_test, y_pred_class)
print("Accuracy: ", accuracy_score(y_test, y_pred_class))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=b,
            yticklabels=b)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()





















