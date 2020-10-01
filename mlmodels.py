from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import dataoperations as dataop
from sklearn.externals import joblib

kfolds = 10 # 90% train, 10% validation
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

clfiers =[LogisticRegression(max_iter=500,C=1,random_state=0), MultinomialNB(alpha = 1000), 
                RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0), MLPClassifier(), LinearSVC(C=1)]
   
names = ["Logistic Regression", "Multinomial NB", "Random Forest", "Neural Networks", "LinearSVM"]

def classifiers(x,entry):
    index = 0
    #the loop to find the chosen algorithm from the option menu
    for name in names:
        if x == name:
            index = names.index(name)
            break
        
    #model is being build with the chosen algorithm
    clf = clfiers[index] 
    clf.fit(dataop.cv_text, dataop.y_train)
    #print("model fitted")
    
    countVec = joblib.load('countvectorizer.pkl') #Previously saved Vectorizer is loaded here.

    #The input entry is being processed to be predicted later
    dataop.lemmatize_words(entry)
    entry_list = [entry]
    entry_array = dataop.np.array(entry_list,dtype = object)
    cv_entry= countVec.transform(entry_array)
    
    # Model is saved here to be used in gui module later
    filename = 'finalized_model.sav'
    joblib.dump(clf, filename)
    
    return cv_entry
     
