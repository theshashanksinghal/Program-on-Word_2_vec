												# GENERATE EMOJIS ACCORDING TO THE SHORT TEXT #

import numpy as np
import emoji
import matplotlib.pyplot as plt

with open("glove.6B.50d.txt","r",encoding='utf-8') as f:
    words=set()
    word_to_vec={}
    for l in f:
        word=l.lower().strip().split()
        words.add(word[0])
        word_to_vec[word[0]]=np.array(word[1:],dtype=np.float64)
    labeling={}
    unlabeling={}
    i=1
    for w in sorted(words):              #for w,i in sorted(word_to_vec):if i do it like this ,it will do everything wrong .
        labeling[w]=i                        # dictinaries are unsorted datatypes..there is no order in dictionaries           
        unlabeling[i]=w
        i+=1
		
###################################################################

def averaging(sen,word_to_vec):
    word=sen.lower().strip().split()
    ave=np.zeros((word_to_vec[word[0]].shape))
    for w in word:
        ave+= word_to_vec[w]
    ave/=len(word)
    return np.asarray(ave,dtype=np.float64)
ave=averaging("Morrocan couscous is my favorite dish", word_to_vec)

###################################################################

def read_csv(file):
    with open(file,'r',encoding='utf-8') as f:
        sentences=[]
        emojis=[]
        for l in f:
            lst=l.strip().split(",")
            sentences.append(lst[0].strip('"'))
            emojis.append(lst[1])
    x=np.asarray(sentences)
    y=np.asarray(emojis,dtype=np.int64)
    return x,y

###################################################################

def predict(x,y,w,b,word_to_vec):
    m=x.shape[0]
    pred=np.zeros((m,1))
    for i in range(m):
        z=np.dot(w,averaging(x[i],word_to_vec))+b
        a=softmax(z)
        pred[i]=np.argmax(a)
    print("accuracy is:",str(np.mean((pred[:]==y.reshape(y.shape[-1],1)[:]))))
    return pred

####################################################################
    
def model(x,y,word_to_vec):
    m=x.shape[0]    #m=shape returns m as a tuple which is (m,)...that's why specify which dimension to take
    classes=5
    glo=50
    num_iter=500
    w=np.random.randn(classes,glo)/np.sqrt(glo)
    b=np.random.randn(5,1)
    for l in range(num_iter):
        for i in range(m):
            y_o=np.zeros((5,1))
            word=averaging(x[i],word_to_vec).reshape(50,1)
            z=np.dot(w,word)+b
            a=softmax(z)
            y_o[int(y[i])]=1   #or i can use - np.eye(5)[y[i]]
            dz=a-y_o
            dw=np.dot(dz,word.T)
            db=dz
            w=w-0.01*dw
            b=b-0.01*db
    return w,b
x_train,y_train=read_csv("train_emoji.csv")
x_test,y_test=read_csv("test_emoji.csv")
w,b=model(x_train,y_train,word_to_vec)
pred=predict(x_test,y_test,w,b,word_to_vec)

###################################################################

													# USING KERAS #

from keras.models import Model
from keras.layers import Input,Dense,Activation,LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

max_len=len(max(x_train,key=len).split())

def labelings(X,labeling):
    m=X.shape[0]
    x_lab=np.zeros((m,max_len))
    for i in range(m):
        words=X[i].lower().split()
        j=0
        for w in words:
            x_lab[i,j]=labeling[w]
            j+=1
    return x_lab
	
def emm_instance(labeling,word_to_vec):
    vocab=len(labeling)+1 #len(word_to_vec.keys())+1
    vec_dim=word_to_vec['the'].shape[0]
    emb_matrix=np.zeros((vocab,vec_dim))
    for i,j in labeling.items():
        emb_matrix[j,:]=word_to_vec[i]
    emb_layer=Embedding(vocab,vec_dim,trainable=False)
    emb_layer.build((None,))
    emb_layer.set_weights([emb_matrix])
    return emb_layer

def emojifier_v2(input_shape,word_to_vec,labeling):
    ip=Input(shape=input_shape,dtype='int32')
    emb_layer=emm_instance(labeling,word_to_vec)
    emb=emb_layer(ip)
    x=LSTM(128,return_sequences=True)(emb)
    x=Dropout(0.5)(x)
    x=LSTM(128,return_sequences=False)(x)
    x=Dropout(0.5)(x)
    x=Dense(5)(x)
    x= Activation('softmax')(x)
    model=Model(inputs=ip,outputs=x)
    return model
	
def emoji_func(num):
    return emoji.emojize(emoji_dict[str(num)],use_aliases=True)
	
emoji_dict={"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
model=emojifier_v2((max_len,),word_to_vec,labeling)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
x_train_lab=labelings(x_train,labeling)
y_train_lab=np.eye(5)[y_train.reshape(-1)]
model.fit(x_train_lab,y_train_lab,epochs=40,batch_size=16,shuffle=True)
x_test_lab=labelings(x_test,labeling)
y_test_lab=np.eye(5)[y_test.reshape(-1)]
loss,acc=model.evaluate(x_test_lab,y_test_lab)
pred=model.predict(x_test_lab)
for i in range(len(x_test)):
    num=np.argmax(pred[i])
    if(num!=y_test[i]):
        print("expect" ,emoji_func(y_test[i])+x_test[i]+"but predicted"+ emoji_func(num))
