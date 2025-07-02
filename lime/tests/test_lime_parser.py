#%%
from __future__ import print_function

import stanza
import random as rand
import numpy as np

#import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import matplotlib

#from lime import lime_text
from sklearn.pipeline import make_pipeline

#from .. import lime_text_parser

from lime.lime_text_parser import LimeTextParserExplainer
from lime.lime_text import LimeTextExplainer
import pickle as pkl

class LimeParserComparison(object):
    def __init__(self, exps=None):
        if not exps:
            self.exps_to_compare = {}
        else:
            self.exps_to_compare = exps

class SavedExplanation(object):
    def __init__(self, name, path, desc=None, exp=None):
        if exp:
            self.name = name
            self.path = path
            self.desc = desc
            self.exp = exp
            pkl.dump(self, path + name)
        else:
            self = pkl.load(path + name)
            return exp
        
    def get_exp(self):
        return self.exp
    
    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.desc
    
    def get_folder(self):
        return self.path
    
    def get_full_path(self):
        return self.path + self.name
    




from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SPAMFILE = r"../smsspamcollection/SMSSpamCollection"

labels = []
texts = []
with open(SPAMFILE, 'r', encoding='utf-8') as file:
    for line in file:
        l, t = line.strip().split('\t')
        if l == 'ham':
            labels.append(0)
        else:
            labels.append(1)
        texts.append(t)


x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=20)


class_names = ['ham', 'spam']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(50, 25), random_state=1)
clf.fit(train_vectors, y_train)
pred = clf.predict(test_vectors)

# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(train_vectors, y_train)
# pred = rf.predict(test_vectors)

sklearn.metrics.f1_score(y_test, pred, average='binary')

c = make_pipeline(vectorizer, clf)

print(c.predict_proba([x_test[0]]))

explainerRan = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="random", random_trees=100)
explainerDep = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="dependency")
explainerCon = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="constituency")
explainerStandard = LimeTextExplainer(class_names=class_names, verbose=True)
#explainer = LimeTextExplainer(class_names=class_names)
idx = 0
while idx < len(x_test):
    print(f"{idx}: {x_test[idx]}")
    idx += 1

idx = 953

# expRand = explainerRan.explain_instance(x_test[idx], c.predict_proba, num_features=5, num_samples=10000)
# expRand2 = explainerRan.explain_instance(x_test[idx], c.predict_proba, num_features=20, num_samples=10000)

# expStd = explainerStandard.explain_instance(x_test[idx], c.predict_proba, num_features=5, num_samples=10000)
# expStd2 = explainerStandard.explain_instance(x_test[idx], c.predict_proba, num_features=20, num_samples=10000)

# exp = explainerDep.explain_instance(x_test[idx], c.predict_proba, num_features=5, num_samples=10000, word_level=True)
# exp2 = explainerDep.explain_instance(x_test[idx], c.predict_proba, num_features=20, num_samples=10000, word_level=True)
exp3 = explainerCon.explain_instance(x_test[idx], c.predict_proba, num_features=5, num_samples=10000)
exp4 = explainerCon.explain_instance(x_test[idx], c.predict_proba, num_features=20, num_samples=10000)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([x_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])

#exp.as_list()

#print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
#tmp = test_vectors[idx].copy()
#tmp[0,vectorizer.vocabulary_['Posting']] = 0
#tmp[0,vectorizer.vocabulary_['Host']] = 0
#print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
#print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


#fig = exp.as_pyplot_figure()
#fig.show()
#exp.show_in_notebook(text=False)


#%%
# expRand.save_to_file('NN2_ran_lime_output.html')
# expRand2.save_to_file('NN2_ran_lime_output2.html')
# exp.save_to_file('NN2_dep_lime_output.html')
# exp2.save_to_file('NN2_dep_lime_output2.html')
exp3.save_to_file('NN1_con_lime_output.html')
exp4.save_to_file('NN1_con_lime_output2.html')
# expStd.save_to_file('NN_std_lime_output.html')
# expStd2.save_to_file('NN_std_lime_output2.html')
#exp.show_in_notebook(text=True)
# %%
