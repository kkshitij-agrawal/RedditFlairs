import pandas as pd #for data arrangement
import praw #for reddit API
import numpy as np 
import matplotlib.pyplot as plt #plotting graph
from sklearn import linear_model #ML
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pymongo

reddit = praw.Reddit(client_id='SvS38h15MT-5sg',\
						client_secret='Na6q8GiTDknIFbyGoI3AgsX_ajE',\
						user_agent='precog_kshitij',\
						username='spacspade',\
						password='qwerty1234')

subreddit = reddit.subreddit('india')

# text = reddit.submission(url="https://www.reddit.com/r/india/comments/cd6h01/flipkart_big_shopping_days_thread_15th18th/")
# print(text.title)


#==================================================
#Training Data Set

train_posts = pd.read_csv("train_data.csv",header=0)
# test_posts = pd.read_csv("test_data.csv",header=0)

# print(train_posts)
# train_posts = subreddit.search('flair_name:"Scheduled"')
test_posts = subreddit.new(limit=100)


topics_dict = { "title":[],\
					"score":[],\
					# "id":[],\
					"comments":[],\
					# "value":[],\
					# "created":[],\
					# "url":[],\
					"flair":[]
					}

test_dict = { "title":[],\
					"score":[],\
					# "id":[],\
					"comments":[],\
					# "value":[],\
					# "created":[],\
					# "url":[],\
					"flair":[]
					}

topics_dict["title"] = train_posts.title
topics_dict["score"] = train_posts.score
topics_dict["comments"] = train_posts.comments
topics_dict["flair"] = train_posts.flair
# topics_dict["value"] = train_posts.value

# print(topics_dict)

# for sub in train_posts:
# 	topics_dict["title"].append(sub.title)
# 	topics_dict["score"].append(sub.score)
# 	# topics_dict["id"].append(sub.id)
# 	topics_dict["comments"].append(sub.num_comments)
# 	# topics_dict["created"].append(sub.created)
# 	# topics_dict["url"].append(sub.url)
	
# 	if(sub.link_flair_text == None):
# 		topics_dict["flair"].append("Non-Political")
# 	else:
# 		topics_dict["flair"].append(sub.link_flair_text)

# test_dict["title"] = test_posts.title
# test_dict["score"] = test_posts.score
# test_dict["comments"] = test_posts.comments
# test_dict["flair"] = test_posts.flair
# test_dict["value"] = test_posts.value


for sub in test_posts:
	test_dict["title"].append(sub.title)
	test_dict["score"].append(sub.score)
	test_dict["comments"].append(sub.num_comments)
	if(sub.link_flair_text == None):
		test_dict["flair"].append("Non-Political")
	else:
		test_dict["flair"].append(sub.link_flair_text)


topics_data = pd.DataFrame(topics_dict) #Created the data frame ie, the data set
test_data = pd.DataFrame(test_dict)

# topics_data.to_csv('reddit_train.csv',index=False) #store the dataset in csv file

categories = list(topics_data.flair) #Storing all categories in a variable
# print(categories)

vector = CountVectorizer() #to the the count of words
X_train = vector.fit_transform(topics_data.title)

vectorizer = TfidfTransformer() #to calculate the weightage
X_train_freq = vectorizer.fit_transform(X_train)

# print(X_train_freq)
clf = MultinomialNB().fit(X_train_freq,categories)

print(test_data.title)
X_test = vector.transform(test_data.title)
X_test_freq = vectorizer.transform(X_test)
predicted = clf.predict(X_test_freq)

for x in predicted:
	print(x)

acc = accuracy_score(test_data.flair,predicted) #Calculating the accuracy
print("Accuracy",acc) 


# ========================================================================
#Graph plot to show the score/comments per category.

temp = pd.read_csv("data.csv")
# print(temp)


plt.scatter(temp.value,topics_data.score,marker='+',color='red')
# plt.show()


# plt.scatter(topics_data.flair,topics_data.score,color='red')
# plt.scatter(topics_data.flair,topics_data.comments,color='blue')

# ========================================================================
# Encoding flair to be predicted by comments and upvotes(score) data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dummies = pd.get_dummies(topics_data.flair)
# print(dummies)

merged = pd.concat([topics_data,dummies],axis='columns')

merged.flair = le.fit_transform(merged.flair)

merged.to_csv('reddit_train.csv',index=False)

final = merged.drop(['flair','Scheduled'],axis='columns')

model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto',max_iter = 2000)
X = final[['comments']].values
# y = temp.value
y = merged[['flair']]

# y = np.reshape(y,(-1,1099))
# print(y.shape)
model.fit(X,y.values.ravel())

# x = np.reshape(test_data.comments,(1,-1))
a = model.predict(test_data[['comments']].values)
print(a)

# res = le.fit_transform(test_data.flair)
# print(res.shape)
# acc2 = model.score(['res'],a)

# ======================================================================

# Taking input as link for test case
flag = 0
while(flag == 0):
	print("Enter a link: ")
	link = input()

	post = reddit.submission(url=link)
	print(post.title)

	X_test = vector.transform([post.title])
	X_test_freq = vectorizer.transform(X_test)
	predicted = clf.predict(X_test_freq)
	print(predicted[0])

	print("continue? y/n")
	ans = input()

	if ans == "n":
		flag = 1


# ======================================================================

# reg = linear_model.LinearRegression()
# reg.fit(topics_data[['score','comments']],topics_data['value'])
# print(reg.coef_)

# # a = np.reshape(test_data.score,(1,-1))
# b = reg.predict(test_data[['score','comments']])
# print(b)

# acc2 = reg.score(test_data.value,b)
# print("Accuracy",acc2)


# plt.plot(topics_data.score,reg.predict(topics_data[['score']]),color='blue')
# plt.plot(topics_data.score,reg.predict(topics_data[['comments']]),color='green')
# # for s in range(len(topics_data.flair)):
# # 	if topics_dict['flair'][s] == 'Photography':
# # 		plt.scatter(topics_dict['score'][s],topics_dict['comments'][s],color='red')
# # 	# else:
# # 	# 	plt.scatter(topics_data['score'][s],topics_data['comments'][s],color='blue')

# plt.show()
