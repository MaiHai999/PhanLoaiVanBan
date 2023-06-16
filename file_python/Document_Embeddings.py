

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set()
from sklearn.svm import LinearSVC
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from sklearn.model_selection import train_test_split
def token_word(s):
    a = []
    for sent in s:
        a1 = []
        h = nltk.word_tokenize(sent)
        for word in h:
            if (word != '.'):
                a1.append(word)
        a.append(a1)
    return a


#lấy dữ liệu và tiền xử lý dữ liệu rồi đua vào huấn luyện
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


#chuẩn hóa dữ liệu cho doc2vect
pre_raw = token_word(X_train)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(pre_raw)]
model = Doc2Vec(documents, vector_size=50, alpha=0.025, min_count=10, dm =1, epochs=100)


#ghi file đã train
fname = "my_doc2vec_model"
model.save(fname)


#khôi phục model và tạo vecto
model = Doc2Vec.load(fname)
train_vectors = [model.infer_vector(nltk.word_tokenize(list_of_tokens), epochs = 50 ) for list_of_tokens in X_train]
test_vectors = [model.infer_vector(nltk.word_tokenize(list_of_tokens), epochs =50) for list_of_tokens in X_test]



#khỏi tạo model phân loại
sv = LinearSVC(class_weight='balanced')
nb = sv.fit(train_vectors, y_train)#train the mode


#dự đoán
y_pred_class = nb.predict(test_vectors)
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



