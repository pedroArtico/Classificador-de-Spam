

import  pandas as pd
from sklearn.model_selection import  cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Youtube02-KatyPerry.csv",index_col="AUTHOR",parse_dates=["DATE"])

count_vect = CountVectorizer()

X_count= count_vect.fit_transform(df.CONTENT)

tfidf_transformer = TfidfTransformer()

X_tfidf = tfidf_transformer.fit_transform(X_count)

d = pd.DataFrame(X_count.A,columns=count_vect.get_feature_names())
d.to_csv("newD.csv")

cross_validation_range = list(range(2,50))

result_LR = []

lr = LogisticRegression()

for i in cross_validation_range:
    scores = cross_val_score(lr, X_tfidf,df.CLASS, cv =i, scoring='accuracy')
    result_LR.append(scores.mean())


k_range = list(range(1,50))

k_scores = []


for i in k_range:
    knn = KNeighborsClassifier ( n_neighbors=i )
    scores = cross_val_score(knn, X_tfidf,df.CLASS, cv =20, scoring='accuracy')
    k_scores.append(scores.mean())


plt.plot(cross_validation_range,result_LR,color='red',marker = '.',markersize = 7,label="Logistic Regression")

plt.plot(k_range,k_scores,color='blue',marker = '.',markersize = 7,label="K-Neighbors")

plt.xlabel('Iteration/n neighbors')
plt.ylabel('Accuracy')

plt.legend()

plt.show()
