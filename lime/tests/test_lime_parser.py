#%%
from __future__ import print_function

import stanza
import random as rand
import numpy as np
#import lime_text_parser
#import stanza.pipeline

#con = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
#con1 = con(sen)

#for sentence in con1.sentences:
#    print(sentence.constituency)

#print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in dep1.sentences for word in sent.words], sep='\n')

#print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
#print(doc)
#print(doc.entities)

#branches = {word.id: (word.head, word.id) for sent in dep1.sentences for word in sent.words}
#ids = [word.text for sent in dep1.sentences for word in sent.words]

#print(*[f"{n}: {branches[n]}" for n in branches.keys()], sep='\n')

#print(f"REMOVING BRANCH {ranbranch}")
    
#stanza.download('en') # <------------------------- ONLY DOES THIS ONCE, FLAGS THE PREVIOUS DOWNLOAD AND STOPS 
#dep = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
#sen = "john saw this movie last weekend, he thought it was not very good, but he still went again"
#dep1 = dep(sen)

#parse_tree, id_dict = organize_parse(dep1)

#ranbranch = rand.choice(range(len(id_dict)))

#print(inverse_removing([7,9], parse_tree, id_dict))


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

#from sklearn.datasets import fetch_openml
#spam = fetch_openml(name="spambase_reproduced")

#print(spam['data'].keys())

'''
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
'''
    #while(not file.closed):
    #    split = file.readline().split("\t")
    #    if len(split) > 1:
    #        l, t = split
    #        labels.append(l)
    #        texts.append(t)
    #    else:
    #        file.close()

from sklearn.model_selection import train_test_split

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


rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, y_train)

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(y_test, pred, average='binary')

c = make_pipeline(vectorizer, rf)

print(c.predict_proba([x_test[0]]))

explainerRan = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="random")
#explainerDep = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="dependency")
#explainerCon = LimeTextParserExplainer(class_names=class_names, verbose=True, parsing_type="constituency")
#explainerStandard = LimeTextExplainer(class_names=class_names, verbose=True)
#explainer = LimeTextExplainer(class_names=class_names)
idx = 0
while idx < len(x_test):
    print(f"{idx}: {x_test[idx]}")
    idx += 1

idx = 953

expRand = explainerRan.explain_instance(x_test[idx], c.predict_proba, num_features=4)
expRand2 = explainerRan.explain_instance(x_test[idx], c.predict_proba, num_features=50)

#expStd = explainerStandard.explain_instance(x_test[idx], c.predict_proba, num_features=4)
#expStd2 = explainerStandard.explain_instance(x_test[idx], c.predict_proba, num_features=50)

#exp = explainerDep.explain_instance(x_test[idx], c.predict_proba, num_features=4)
#exp2 = explainerDep.explain_instance(x_test[idx], c.predict_proba, num_features=50)
#exp3 = explainerCon.explain_instance(x_test[idx], c.predict_proba, num_features=4)
#exp4 = explainerCon.explain_instance(x_test[idx], c.predict_proba, num_features=50)
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
expRand.save_to_file('ran_lime_output.html')
expRand2.save_to_file('ran_lime_output2.html')
# exp.save_to_file('dep_lime_output.html')
# exp2.save_to_file('dep_lime_output2.html')
# exp3.save_to_file('con_lime_output.html')
# exp4.save_to_file('con_lime_output2.html')
# expStd.save_to_file('std_lime_output.html')
# expStd2.save_to_file('std_lime_output2.html')
#exp.show_in_notebook(text=True)
# %%
