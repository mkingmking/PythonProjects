from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np



texts = ["love it", "like love", "like it ", " love love love it " , "disgust it disgust it" , "hate it" , "hate", "disgust", "hate hate hate hate", "disgust disgust"] 


y_texts = [1,1,1,1,0,0,0,0,0,0]


cv = CountVectorizer()

cv_fit = cv.fit_transform(texts)


X_train = cv_fit.toarray()

print(X_train)


i2w = {v:k for k,v in cv.vocabulary_.items()}

print(i2w)


#creating dataframe

df = pd.DataFrame(X_train, columns = [i2w[i]  for i in range (len(i2w))])

df['LABEL' ] = y_texts

print(df)



###training the model


priors = df["LABEL"].value_counts() / len(df)

#rename index labels

priors = priors.rename({0:'class_0' , 1: 'class_1'} )

print(priors)


## the number of times term xi appears in class y = 0


counts0 = df[df["LABEL"] == 0].sum(axis=0)

print(counts0)

counts1 = df[df["LABEL"] == 1].sum(axis=0)

print(counts1)


#Concatenate along axis 1 (columns)

result = pd.concat([counts0, counts1], axis=1)
result.columns = ['class_0', 'class_1']

result = result.drop('LABEL')

print(result)

smoothed_result = result + 1

print(smoothed_result)


total_count = smoothed_result.sum(axis=1)


likelihood = smoothed_result.copy()

likelihood['class_0'] = smoothed_result['class_0'] / total_count

likelihood['class_1'] = smoothed_result['class_1'] / total_count

print(likelihood)

log_likelihood = np.log(likelihood)

log_priors = np.log(priors)


## inference
x = ['love like love', 'hate hate disgust it', 'missing word']
cv_fit_test = cv.transform(x)
x = cv_fit_test.toarray()



print(x.dot(log_likelihood.values) + log_priors.values)

print(np.argmax(x.dot(log_likelihood.values)  + log_priors.values, axis = 1))







