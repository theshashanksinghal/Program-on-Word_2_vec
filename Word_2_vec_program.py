							# WORD_2_VEC # SIMILARITY # ANALOGY # DEBIASING # EQUALIZATION
							
#since this method consumes large memory that's wht we won't use it
'''data=open("glove.6B.50d.txt",encoding='utf-8').readlines()
words=set()
word_2_vec={}
for w in data:
    word=w.strip().split()
    words.add(word[0])
    word_2_vec[word[0]]=np.array(word[1:],dtype=np.float64)'''


with open("glove.6B.50d.txt","r",encoding='utf-8') as f:
    words=set()
    word_2_vec={}
    for l in f:
        word=l.strip().split()
        words.add(word[0])
        word_2_vec[word[0]]=np.array(word[1:],dtype=np.float64)  
		
####################################################

												# COSINE SIMILARITY #
		
def cos_sim(word1,word2,word_2_vec):
    em1=word_2_vec[word1]
    em2=word_2_vec[word2]
    dot=np.dot(em1,em2)
    mod_em1=np.sqrt(np.dot(em1.T,em1))#or you can do this also - np.sqrt(np.sum(v*v))
    mod_em2=np.sqrt(np.dot(em2.T,em2))
    sim=dot/(mod_em1*mod_em2)
    return sim

a=cos_sim('father','man',word_2_vec)

####################################################

													# WORD_ANALOGY #

def cos_sim1(em1,em2,word_2_vec):
    dot=np.dot(em1,em2)
    mod_em1=np.sqrt(np.sum(em1*em1))#or you can do this also - np.sqrt(np.sum(v*v))
    mod_em2=np.sqrt(np.sum(em2*em2))
    sim=dot/(mod_em1*mod_em2)
    return sim

def analogy(word1,word2,word3,word_2_vec):
    word1,word2,word3=word1.lower(),word2.lower(),word3.lower()
    em1=word_2_vec[word1]
    em2=word_2_vec[word2] 
    em3=word_2_vec[word3]
    words=word_2_vec.keys()
    def_sim=-999
    for word in words:
        if word in [word1,word2,word3]:#if i dont write this i'll get this error-
            continue                   # __main__:5: RuntimeWarning: invalid value encountered in double_scalars
        similarity=cos_sim1((em1-em2),(em3-word_2_vec[word]),word_2_vec)
        if similarity>def_sim:
            def_sim=similarity
            req_word=word
    return req_word
a=analogy('boy','girl','father',word_2_vec)

#######################################################

														# DEBIASING #
														
g=(word_2_vec['mother']-word_2_vec['father']+word_2_vec['girl']-word_2_vec['boy']+word_2_vec['female']-word_2_vec['male']+word_2_vec['grandmother']-word_2_vec['grandfather']+word_2_vec['women']-word_2_vec['man'])/5
#notice that girls name have +ive similarity and boys have -tive similarity...and thats what was expected
l=['john','marie','sophie','ronaldo','rahul','priya']
for i in l:
    print(i,cos_sim(word_2_vec[i],g,word_2_vec))
#notice that these profession that are not gender specific shows gender biasing...which is not good
s=['fashion','engineer','warrior','gun','science','computer','art','literature','babysitter','nurse','doctor']
for i in s:
    print(i,cos_sim(word_2_vec[i],g,word_2_vec))
def neutralize(word,word_2_vec):
    g=(word_2_vec['mother']-word_2_vec['father']+word_2_vec['girl']-word_2_vec['boy']+word_2_vec['female']-word_2_vec['male']+word_2_vec['grandmother']-word_2_vec['grandfather']+word_2_vec['women']-word_2_vec['man'])/5
    word=word.lower()
    em=word_2_vec[word]
    bias=(np.dot(em,g)/np.dot(g.T,g))*g
    debias=em-bias
    return debias
#now you will notice that you will get a very large NEGATIVE value...signifies that there is similarity between the word and the vector that represent wonam specific gender(female-male)
print("cos_sim with bias- engineer",cos_sim1(word_2_vec['engineer'],g,word_2_vec))
print("cos_sim without bias- engineer",cos_sim1(neutralize('engineer',word_2_vec),g,word_2_vec))

#######################################################

													# EQALIZATION #
													
def equalize(words,g,word_2_vec):
    word1,word2=words
    em1,em2=word_2_vec[word1],word_2_vec[word2]
    ave=(em1+em2)/2
    ave_b=(np.dot(ave,g)/np.sum(np.dot(g.T,g)))*g
    ave_u=ave-ave_b
    em1_b=(np.dot(em1,g)/np.sum(np.dot(g.T,g)))*g
    em2_b=(np.dot(em2,g)/np.sum(np.dot(g.T,g)))*g
    corre_em1_b=(np.sqrt(abs(1-np.dot(ave_u.T,ave_u))))*(em1_b-ave_b)/np.linalg.norm((em1_b-ave_b))#np.sqrt(np.dot((em1_b-ave_b).T,(em1_b-ave_b)))
    corre_em2_b=(np.sqrt(abs(1-np.dot(ave_u.T,ave_u))))*(em2_b-ave_b)/np.linalg.norm((em2_b-ave_b))#np.sqrt(np.dot((em1_b-ave_b).T,(em1_b-ave_b)))
    e1=corre_em1_b + ave_u
    e2=corre_em2_b + ave_u
    return e1,e2

g=word_2_vec['female']-word_2_vec['male']
e1,e2=equalize(('grandfather','grandmother'),g,word_2_vec)
							
