import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np


with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

# Example:
data_point = data[3000] # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))


#Dividing data
train_data = data[:10000] #data[:10000]
print("Train :" + str(len(train_data)))

vd_data = data[10000:11000]
print("vd :" + str(len(vd_data)))


test_data = data[11000:]
print("Test :" + str(len(test_data)))



#helper fun1
#convert a text to a list of words
def text_pross(string):
    str_l = string.lower() 
    words = str_l.split( )
    
    return words


##helper fun2
# return a list of 160 most common words
def common_words():
    sum_text = str()
    for data_point in train_data:
        text = data_point['text']
        sum_text = sum_text + text
        
    words_list = text_pross(sum_text) 
    most_occur = Counter(words_list).most_common(10)  #160
    
   # have to convert vocb_list to a words only no numbers  
   #type = list / list of tuple elem  
    vocb_list = [] 
    for elem in most_occur:
            word = elem[0]
            vocb_list.append(word)
            
    return vocb_list    


#helper fun3
#return vector x_counts  (160x1)  
#def compare(vocb_list, data_point):
#     #size 160x1
#    
#    x_counts = np.zeros((160,1), dtype=np.int)
#    #print(x_counts)
#    text = data_point['text']
#    words = text_pross(text)
#    for word in words:
#        for vocb in vocb_list:
#            if vocb == word:
#                #have to find the index of the word
#                indx = vocb_list.index(word)
#                x_counts[indx] += 1
#               
#                
#    return x_counts


def compare(vocb_list, data_point):
     #size 160x1
    temp=0
    x_counts = np.zeros((166,), dtype=np.int)
    x_counts[0]=1
    #print(x_counts)
    text = data_point['text']
    words = text_pross(text)
    for word in words:
        for vocb in vocb_list:
            if vocb == word:
                
                #have to find the index of the word
                indx = vocb_list.index(word)
                x_counts[indx+1] += 1
    
             
    temp = np.count_nonzero(x_counts)          
    x_counts[161] = bool_num(data_point['is_root']) #x01
    x_counts[162] = data_point['controversiality'] #x02
    x_counts[163] = data_point['children'] #x03  
    
    x_extra_feat1 = len(text_pross(data_point['text']))
    x_counts[164] = x_extra_feat1 #x03 
    x_counts[165] = temp - 1
    
    x_counts = x_counts.T 
      
    return x_counts


def X_stack(data_set):
    
    X = np.zeros((0,166), dtype=np.int)
    vocb_list = common_words()
    for data_point in data_set:
        x1 = compare(vocb_list, data_point)
        X=np.vstack([X,x1])
        
    return X   


print(X_stack(train_data))


##testing fun1&2&3
#vocb_list = common_words()
#print(vocb_list)
#x_count_7000 = compare(vocb_list,data_point)
#print(x_count_7000)



def bool_num(boolean):
    if boolean == False:
        root = 0
    else:
        root = 1
        
    return root



#need y= number of examples x 1 = popularity score
#need x= number of examples x number of features(164)
#features = (bias term, children, controversiality, is_root, 
#             word1_frequency, word2_frequency, ..., word160_frequency,
#              extra_feature1, extra_feature2)

#helper fun4
#input is data set
#output = y_matrix of popularity_score (nX1) 
def yMatrix(data_set):
    n = len(data_set)
    y_matrix = np.zeros((n,1), dtype=np.float)
    for data_point in data_set:
        y_i = data_point['popularity_score']
        indx = data_set.index(data_point)
        y_matrix[indx] = y_i
        
    return y_matrix

#testing fun4
y_matrix = yMatrix(train_data)            
print(y_matrix.shape)   
print(y_matrix)   

#helper fun4
#input is data set and number of features
# for now no extra features
#output = X_matrix  (nXm) 
#def xMatrix(data_set, num_f):
#    vocb_list = common_words()
#    n = len(data_set) 
#    x_matrix = np.zeros((n,num_f), dtype=np.float)
#    #adding bias term
#    ones = np.ones([x_matrix.shape[0],1])
#    print(ones.shape)
#    print(x_matrix.shape)
#    x_matrix = np.concatenate((ones,x_matrix),axis=1)
#    
#   
#    for data_point in data_set: #i
#        j = 1
#        indx = data_set.index(data_point)
#        #for j in range(num_f):  #j
##        x_matrix[indx, j] = bool_num(data_point['is_root']) #x01
##        j+=1
##        x_matrix[indx, j] = data_point['controversiality'] #x02
##        j+=1
##        x_matrix[indx, j] = data_point['children'] #x03
##        j+1
#        
#        x1= bool_num(data_point['is_root']) #x01
#        x2= data_point['controversiality'] #x02
#        x3 = data_point['children'] #x03
#        
#        x_count = compare(vocb_list, data_point)
#        x_count.insert(0, x3) 
#        x_count.insert(0, x2) 
#        x_count.insert(0, x1) 
#        
#        x_matrix = np.concatenate((x_count,x_matrix),axis=0)
#        
#        x_matrix = np.concatenate((ones,x_matrix),axis=1)
#        
##            if j == (num_f - 1):
##                break
#             
#    return x_matrix
##
#x_matrix = xMatrix(train_data,164)            
#print(x_matrix.shape)   
#print(x_matrix)

    
# have to convert vocb_list to a words only no numbers  
##text fun2
#vocb_list = common_words()
#   #type = list  spyder
# #list of tuple elem  
#word_list = [] 
#for elem in vocb_list:
#        word = elem[0]
#        word_list.append(word)
    
    
#print(word_list)

#def count_words(string):
#    
#    #counts_all = []
#    counts = dict()
#    words = text_pross(string)
#    
#    #counts = Counter(words)    
#    for word in words:
#        if word in counts:
#            counts[word] += 1
#        else:
#            counts[word] = 1 
#                     
#    return counts
#
#counts = count_words(data_point['text'])
#print(counts)
#
#
#
      
##helper fun3 
#def most_common():  
#  counts_all = dict() 
#   
#  common = dict()
#  for data_point in train_data:
#      counts = count_words(data_point['text'])
#      #print("counts type is:" + str(type(counts)))
##      for key,value in counts.items():
##          if key in counts.keys():
##              value += counts.get(key)
##              
##          else:
##              counts[key] = value
#              
#      counts_all.update(counts)         
#              
#              
#      #counts_all.update(counts)
#                  
#            
#  most_occur = Counter(counts_all).most_common(160) 
#  #most_occur = sorted(counts_all.keys())
#  #most_occur = sorted(counts_all.items(), key=itemgetter(1))
#  #print(most_occur) 
#  print(most_occur)
#  return  most_occur


# should be a 160x1 list or vector
#common_words = most_common()


##helper fun4
#def compare(common_words, string):
#     #size 160x1
#    x_counts = np.zeros((160,1), dtype=np.int)
#    #print(x_counts)
#    
#    for vocb in string:
#        for word in common_words:
#            if vocb == word:
#                #have to find the index of the word
#                indx = common_words.index(word)
#                x_counts[indx] += 1
#               
#            
#            
#    return x_counts        
#            
#
#
##test  compare method      
#words = ["the", "wheel", "what", "if", "is", "out"]   
#text = "what OUT if the world" 
##vocb_list = ocuur()

#print(compare(vocb_list,data_point['text']))


#for info_name, info_value in counts.items():
#   print(info_name + " : " + str(info_value))
   
   
#helper3 finding the 160 most commen words   
#def most_commen(string):
#    counts = dic()
