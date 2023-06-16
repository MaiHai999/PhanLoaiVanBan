

import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib
from tensorflow.keras.layers import Embedding
from tensorflow import keras
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM


##lấy dữ liệu và tiền xử lý dữ liệu rồi đua vào huấn luyện
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
    lable_contruction.append(1)

raw_economics = nltk.sent_tokenize(raw_economics)
lable_economics = []
for i in range(len(raw_economics)):
    lable_economics.append(2)


raw_food = nltk.sent_tokenize(raw_food)
lable_food = []
for i in range(len(raw_food)):
    lable_economics.append(3)



raw_history = nltk.sent_tokenize(raw_history)
lable_history = []
for i in range(len(raw_history)):
    lable_history.append(4)



raw_medican = nltk.sent_tokenize(raw_medican)
lable_medican = []
for i in range(len(raw_medican)):
    lable_medican.append(5)



#tạo tập train và tập test
x = raw_medican + raw_food + raw_history + raw_contruction + raw_economics
y = lable_medican + lable_food + lable_history + lable_contruction + lable_economics


#phân chia tập test và tập train
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)


#tạo từ điển và mã hóa từng câu
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index


#biến đổ để cho tất cả về cùng một chiều dữ liệu và coi như đầu vào của dữ liệu
trainvalid_data = pad_sequences(train_sequences, maxlen = 100 )
test_data = pad_sequences(test_sequences, maxlen = 100)
trainvalid_labels = to_categorical(np.asarray(y_train))
test_labels = to_categorical(np.asarray(y_test))

#phân chia để cho vào model
x_train1 ,x_val, y_train1 ,y_val =  train_test_split(trainvalid_data, trainvalid_labels, random_state=1)


##train thủ công tự train phần embedding bằng LSTM
#khôi phục model embedding
filename = 'C:\\Users\\HP\\PycharmProjects\\embeding_text_classcifition\\vectors_100.kv'
model1 = joblib.load(filename)


#tạo một ma trận embeding trong từ điển
num_words = 13685
embedding_matrix = np.zeros((num_words, 100))
for word, i in word_index.items():
    if word in model1.wv:
        embedding_vector = model1.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



#hàm chuyển đổi câu thành vecto
embedding_layer = Embedding(num_words, 100, weights  = [embedding_matrix],
 input_length=100,
 trainable=False)



#xây dựng model
print("Defining and training an LSTM model, training embedding layer on the fly")
rnnmodel = Sequential()
rnnmodel.add(embedding_layer)
rnnmodel.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
rnnmodel.add(Dense(6, activation='sigmoid'))
rnnmodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Training the RNN')
history = rnnmodel.fit(x_train1, y_train1,batch_size=100,epochs=5 ,validation_data=(x_val, y_val))
score, acc = rnnmodel.evaluate(test_data, test_labels,batch_size=32)
print('Test accuracy with RNN:', acc)

#vẽ sơ đồ
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

#lưu model
rnnmodel.save("my_model_RNN_text_class.h5")




