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

from lime.lime_text_parser import LimeTextParserExplainer, SavedExplanation
from lime.lime_text import LimeTextExplainer
import dill as pkl
import os

class LimeParserComparison(object):
    def __init__(self, exps=None):
        if not exps:
            self.exps_to_compare = {}
        else:
            self.exps_to_compare = exps


        
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
from tqdm import tqdm

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

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, y_train)
pred = rf.predict(test_vectors)

sklearn.metrics.f1_score(y_test, pred, average='binary')

r = make_pipeline(vectorizer, rf)
m = make_pipeline(vectorizer, clf)

print(r.predict_proba([x_test[0]]))

#explainerRan = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="random", random_trees=100)
#explainerDep = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="dependency")
#explainerCon = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="constituency")
#explainerStandard = LimeTextExplainer(class_names=class_names, verbose=True)
#explainer = LimeTextExplainer(class_names=class_names)
idx = 0
while idx < len(x_test):
    print(f"{idx}: {x_test[idx]}")
    idx += 1

idx = 953
print('Document id: %d' % idx)
print('Probability(christian) =', r.predict_proba([x_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


EXPL_PATH = r"./saved_explanations/"
HTML_PATH = r"./HTML_results/"

def run_all_explainers(models, class_names, parameter_sets, instances, save=False, descriptions=None, path=None):
    explainerRan = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="random")
    explainerDep = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="dependency")
    explainerCon = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="constituency")
    explainerStd = LimeTextExplainer(class_names=class_names, verbose=True)
    explanations = []

    def save_name(j, m, i, desc):
        name = desc["disting"] + "_" + desc["parses"][j] + "_" + desc["models"][m] + "_" + str(i)
        return name

    def save_desc(j, m, i, fts, smp, desc, msk=None, rnd=None, wrd=None):
        name=save_name(j, m, i, desc=desc)
        description=name
        description+="\n"+"Features:\t"+str(fts)+"\nSamples:\t"+str(smp)
        if msk != None:
            description += "\nMask Method:\t" + str(msk)
        if rnd != None:
            description += "\nRandom Trees:\t" + str(rnd)
        if wrd != None:
            description += "\nWord Level:\t" + str(wrd)
        return name, description

    for m, model in enumerate(tqdm(models, "Model")):
        for i, inst in enumerate(tqdm(instances, "Instance")):
            #prediction = model([inst])
            for pset in tqdm(parameter_sets, "Parameter Set"):
                (num_feats, num_samples, mask_method, num_rand_trees, word_level) = pset
                explanations.append(explainerDep.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                  mask_method=mask_method, word_level=word_level))
                explanations.append(explainerCon.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                  mask_method=mask_method))
                explanations.append(explainerRan.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                  random_trees=num_rand_trees, mask_method=mask_method))
                explanations.append(explainerStd.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples))

                if save:
                    name, desc = save_desc(0,m,i,num_feats,num_samples,descriptions,mask_method,wrd=word_level)
                    SavedExplanation(name, path, desc, explanations[-4])
                    name, desc = save_desc(1,m,i,num_feats,num_samples,descriptions,mask_method)
                    SavedExplanation(name, path, desc, explanations[-3])
                    name, desc = save_desc(2,m,i,num_feats,num_samples,descriptions,mask_method,rnd=num_rand_trees)
                    SavedExplanation(name, path, desc, explanations[-2])
                    name, desc = save_desc(3,m,i,num_feats,num_samples,descriptions)
                    SavedExplanation(name, path, desc, explanations[-1])

    return explanations

def load_explanations(descs, path):
    explanations = []
    for parse in descs["parses"]:
        for model in descs["models"]:
            i = 0
            while os.path.exists(path+descs["disting"]+"_"+parse+"_"+model+"_"+str(i)+".pkl"):
                print("Found\t"+descs["disting"]+"_"+parse+"_"+model+"_"+str(i)+".pkl")
                explanations.append(SavedExplanation(descs["disting"]+"_"+parse+"_"+model+"_"+str(i),path))
                i += 1

    return explanations
    
parameter_sets = [(10, 1000, 1, 5, True), 
                  (20, 1000, 1, 5, True), 
                  (10, 1000, 2, 5, True), 
                  (20, 1000, 2, 5, True)]

descs = {
    "models": ["RF_500", "MLP_(50,25)"],
    "parses": ["Dep", "Con", "Ran", "Std"],
    "disting": "Test1"
}

run_all_explainers([r.predict_proba, m.predict_proba], class_names, parameter_sets, [x_test[idx]], save=True, descriptions=descs, path=EXPL_PATH)
loaded = load_explanations(descs, EXPL_PATH)

for ep in loaded:
    print(str(ep.get_desc()) + "\n")

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

def exp_cos_similarity(exp1, exp2):
    exp1 = [x[1] for x in exp1]
    exp2 = [x[1] for x in exp2]


# SavedExplanation("NN1_Con_2_msk1", EXPL_PATH, "MLP=(50,25), num_features=20, num_samples=10000", exp3)
# SavedExplanation("NN1_Con_2_msk2", EXPL_PATH, "MLP=(50,25), num_features=20, num_samples=10000", exp4)
#exp3 = SavedExplanation("NN1_Con_1", EXPL_PATH).get_exp()


#%%
# expRand.save_to_file('NN2_ran_lime_output.html')
# expRand2.save_to_file('NN2_ran_lime_output2.html')
# exp.save_to_file('NN2_dep_lime_output.html')
# exp2.save_to_file('NN2_dep_lime_output2.html')
#exp3.save_to_file('NN1_Con_2_msk1.html')
#exp4.save_to_file('NN1_Con_2_msk2.html')
# expStd.save_to_file('NN_std_lime_output.html')
# expStd2.save_to_file('NN_std_lime_output2.html')
#exp.show_in_notebook(text=True)
# %%
