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

from lime.lime_text_parser import LimeTextParserExplainer, SavedExplanation, IndexedStringParsed
from lime.lime_text import LimeTextExplainer
import dill as pkl
import os

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin

#///////ADD BERT COMPARISON WITH TENSORFLOW//////////

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
import re
import sys



def clear_lines(n):
# Move up and clear n lines
    for _ in range(n):
        sys.stdout.write("\033[F")  # move cursor up
        sys.stdout.write("\033[2K") # clear line
    sys.stdout.flush()

EXPL_PATH = r"./saved_explanations/"
MODEL_PATH = r"./saved_models/"
HTML_PATH = r"./HTML_results/"

def run_all_explainers(models, class_names, parameter_sets, instances, save=False, descriptions=None, path=None, 
                       skip_existing=False, just_desc=False):

    def save_name(j, m, i, p, desc):
        name = desc["disting"] + "_" + desc["parses"][j] + "_" + desc["models"][m] + "_" + str(p) + "_" + str(i)
        return name

    def save_desc(j, m, p, i, fts, smp, desc, msk=None, rnd=None, wrd=None):
        name=save_name(j, m, i, p, desc=desc)
        description=name
        description += "\nModel:\t\t" + desc["models"][m]
        description+="\nFeatures:\t"+str(fts)+"\nSamples:\t"+str(smp)
        if msk != None:
            description += "\nMask Method:\t" + str(msk)
        if rnd != None:
            description += "\nRandom Trees:\t" + str(rnd)
        if wrd != None:
            description += "\nWord Level:\t" + str(wrd)
        return name, description

    explainerRan = LimeTextParserExplainer(class_names=class_names, verbose=False, parsing_type="random")
    clear_lines(21)
    explainerDep = LimeTextParserExplainer(class_names=class_names, verbose=False, parsing_type="dependency")
    clear_lines(26)
    explainerCon = LimeTextParserExplainer(class_names=class_names, verbose=False, parsing_type="constituency")
    clear_lines(28)
    explainerStd = LimeTextExplainer(class_names=class_names, verbose=False)
    explanations = []
    
    tot_insts = len(instances)
    tot_params = len(parameter_sets)
    tot_models = len(models)
    total = tot_insts*tot_params*tot_models
    progress = 0

    for m, model in enumerate(models):
        for i, inst in enumerate(instances):
            #prediction = model([inst])
            for p, pset in enumerate(parameter_sets):
                progress = m*tot_insts*tot_params + i*tot_params + p + 1
                print(f"\n[  {int((1000 * progress) / total) / 10.0}%  |  {progress}/{total}  ]\n\n", end='', flush=True)
                (num_feats, num_samples, mask_method, num_rand_trees, word_level) = pset

                if just_desc:
                    skip_existing = True 
    
                name = save_desc(0,m,p,i,num_feats,num_samples,descriptions,mask_method,wrd=word_level)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerDep.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                    mask_method=mask_method, word_level=word_level))
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())

                
                name = save_desc(1,m,p,i,num_feats,num_samples,descriptions,mask_method)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerCon.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                    mask_method=mask_method))
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    
                name = save_desc(2,m,p,i,num_feats,num_samples,descriptions,mask_method,rnd=num_rand_trees)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerRan.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples, 
                                                                    random_trees=num_rand_trees, mask_method=mask_method))
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    
                name = save_desc(3,m,p,i,num_feats,num_samples,descriptions)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerStd.explain_instance(inst, model, num_features=num_feats, num_samples=num_samples))
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())

                if just_desc:
                    skip_existing = False

                if save:    
                    name, desc = save_desc(0,m,p,i,num_feats,num_samples,descriptions,mask_method,wrd=word_level)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-4])
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(1,m,p,i,num_feats,num_samples,descriptions,mask_method)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-3])
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(2,m,p,i,num_feats,num_samples,descriptions,mask_method,rnd=num_rand_trees)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-2])
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(3,m,p,i,num_feats,num_samples,descriptions)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-1])
                    else:
                        print("\n" + name + " exists" + "\n")
                
                clear_lines(15)

    return explanations

def load_explanations(descs, path, specific=False):
    
    explanations = []

    if specific:
        patterns = []

        for parse in descs["parses"]:
            for model in descs["models"]:
                disting = descs["disting"]
                patterns.append(re.compile(rf"{re.escape(disting)}_{re.escape(parse)}_{re.escape(model)}.*\.pkl$"))
        
        for filename in os.listdir(path):
            for pattern in patterns:
                if pattern.match(filename):
                    explanations.append(SavedExplanation(filename, path))
                    break

    else:
        disting = descs["disting"]
        pattern = re.compile(rf"{re.escape(disting)}.*\.pkl$")
        for filename in os.listdir(path):
            if pattern.match(filename):
                explanations.append(SavedExplanation(filename, path))

    return explanations

def get_model(desc):
    lines = desc.split("\n")
    for line in lines:
        if line.startswith("Model"):
            x = line.split("\t")
            return x[-1]
        
def get_explr(desc):
    first_line = desc.split("\n")[0].split("_")
    explr = first_line[1]
    return explr

def exp_cos_similarity(exp1, exp2):
    exp1 = [x[1] for x in exp1]
    exp2 = [x[1] for x in exp2]
    return cos_similarity(exp1, exp2)
    
def cos_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    try:
        return dot / (norm1 * norm2)
    except:
        return 0

class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        vectors = []
        for sentence in X:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True,
                                    padding=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # CLS token vector (shape: [1, hidden_dim])
            cls_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vectors.append(cls_vector[0])
        return np.vstack(vectors)


def save_bert_vecs(dataset, filename, path):
    try:
        texts, labels = dataset
    except:
        print("ERROR: No data provided!")
        return None
    bert_sens = []
    save_data = []
    bert_vectorizer = BERTVectorizer()
    bert_sens = bert_vectorizer.transform(texts)
    save_data = (texts, bert_sens, labels)
    with open(path + filename + ".pkl", "wb+") as file:
        pkl.dump(save_data, file)

    return save_data

def load_bert_vecs(filename, path):
    with open(path + filename + ".pkl", "rb") as file:
        return pkl.load(file)


def get_data(filename, path, data=None, split_ratio=0.2, random_state=20, text_too=False):
    try:
        data = load_bert_vecs(filename, path)
        print("File found...")
    except:
        print("No file found, computing BERT vectors...")
        data = save_bert_vecs(data, filename, path)
    zip_data = list(zip(data[0], data[1], data[2]))
    train_data, test_data = train_test_split(zip_data, test_size=split_ratio, random_state=random_state)
    train_sens, train_bert, y_train = zip(*train_data)
    test_sens, test_bert, y_test = zip(*test_data)
    clear_lines(1)
    print("Recieved BERT vectors")
    if not text_too:
        return (train_bert, test_bert, y_train, y_test)
    else:
        return (train_sens, test_sens, train_bert, test_bert, y_train, y_test)
    

def influence_sz(exp, label=1):
    sum_influences = []
    weighted_sum_infl = []
    relation_distance = []
    total = 0
    num_words = exp.get_exp().domain_mapper.indexed_string.num_words()
    # print(exp.get_local_exp())
    # print(exp.get_text())
    # print(exp.get_tokens())
    if len(exp.get_local_exp()) == 0:
        print(exp.get_desc())
    # exp.get_exp().save_to_file("INFLUENCETEST.html")

    if not exp.data["is_standard"]:
        local_exp = exp.data["exp_data"]["domain_data"]["all_exps"][label]
        ids = [x[0] for x in local_exp]

        dependence_chunks = [[] for _ in range(int(max(ids)/num_words) + 1)]
        id_chunks = [[] for _ in range(int(max(ids)/num_words))]
        weighted_dep_chunks = [0 for _ in range(int(max(ids)/num_words))]
        for i in local_exp:
            dependence_chunks[int(i[0]/num_words)].append(i)
        for i, chunk in enumerate(dependence_chunks[1:]):
            weighted_dep_chunks[i] = chunk[0][1]
            id_chunks[i] = [x[0] for x in chunk]

        sum_influences = [len(dep) for dep in id_chunks]
        weighted_sum_infl = [sum_ * abs(weighted_dep_chunks[i]) for i, sum_ in enumerate(sum_influences)]
        relation_distance = [abs(max(id) - min(id)) for id in id_chunks]
        weighted_rel_dist = [dist * abs(weighted_dep_chunks[i]) for i, dist in enumerate(relation_distance)]
        total = len(id_chunks)

    else:
        get_exp = exp.get_exp().local_exp[label]
        sum_influences = [1 for _ in get_exp]
        weighted_sum_infl = [x[1] for x in get_exp]
        relation_distance = sum_influences
        weighted_rel_dist = weighted_sum_infl
        total = len(get_exp)

    return (sum(sum_influences) / total, 
            sum(weighted_sum_infl) / total,
            sum(relation_distance) / total,
            sum(weighted_rel_dist) / total,
            num_words)


def avg_influence_sz(exp_arr):
    avg_sz = 0
    wgh_sz = 0
    dist_avg = 0
    wgh_dst = 0
    word_sum = 0
    for exp in exp_arr:
        infl_sz = influence_sz(exp)
        avg_sz += infl_sz[0]
        wgh_sz += infl_sz[1]
        dist_avg += infl_sz[2]
        wgh_dst += infl_sz[3]
        word_sum += infl_sz[4]
    avg_sz = avg_sz / word_sum
    wgh_sz = wgh_sz / word_sum
    dist_avg = dist_avg / word_sum
    wgh_dst = wgh_dst / word_sum
    return (avg_sz, wgh_sz, dist_avg, wgh_dst)


def model_save_name(params, dataset=None, ext=True):
    filename = ''
    if dataset != None:
        filename += dataset + "-"
    filename += params[0] + "-" + params[1]
    if params[0] == "rf":
        filename += "-" + str(params[2])
    elif params[0] == "mlp":
        for l in params[2]:
            filename += "-" + str(l)
    if ext:
        return filename + ".pkl"
    else:
        return filename


def train_models(all_model_params, dataset=None):
    print("Training Models...")
    models_trained = []
    bert_vectorizer = BERTVectorizer()
    for p in all_model_params:
        m = []
        if p[0] == "rf":
            m = sklearn.ensemble.RandomForestClassifier(n_estimators=p[2])
        elif p[0] == "mlp":
            m = MLPClassifier(solver="lbfgs", alpha=1e-5,
                            hidden_layer_sizes=tuple(p[2]), random_state=1)
        if p[1] == "i":
            m.fit(train_vectors, y_train)
            models_trained.append(make_pipeline(vectorizer, m).predict_proba)
        elif p[1] == "b":
            m.fit(bert_train, y_train)
            models_trained.append(make_pipeline(bert_vectorizer, m).predict_proba)
        with open(MODEL_PATH + model_save_name(p, dataset), "wb") as file:
            pkl.dump(models_trained[-1], file)
    clear_lines(1)
    print("Models trained")
    return models_trained

def load_models(all_model_params, dataset=None):
    print("Loading models...")
    models_loaded = []
    for p in all_model_params:
        with open(MODEL_PATH + model_save_name(p, dataset), "rb") as file:
            models_loaded.append(pkl.load(file))
    clear_lines(1)
    print("Models loaded")
    return models_loaded


BERT_FOLD = r"./bert_data/"
SPAM_DATA = "spam_ds"
SEM_DATA = "sem_ds"
IMDB_DATA = "imdb_ds"
SPAMFILE = r"./smsspamcollection/SMSSpamCollection"
SEMFILE = r"./sentiment_sens/"
SEMFILE_1 = "amazon_cells_labelled.txt"
SEMFILE_2 = "imdb_labelled.txt"
SEMFILE_3 = "yelp_labelled.txt"
IMDB = r"./aclImdb/"
IMDB_TT = ["test/", "train/"]
IMDB_NP = ["neg/", "pos/"]
IMDB_COMP = IMDB + "imdb_compiled.txt"

# texts = []
# labels = []
# for i in range(2):
#     print(f"i = {i}")
#     for j in range(2):
#         print(f"j = {j}")
#         for filename in os.listdir(IMDB + IMDB_TT[i] + IMDB_NP[j]):
#             with open(IMDB + IMDB_TT[i] + IMDB_NP[j] + filename, "r", encoding="utf-8") as file:
#                 texts.append(file.readline().replace("<br />", " "))
#             labels.append(j)

# with open(IMDB + "imdb_compiled.txt", "w+", encoding='utf-8') as file:
#     for l, t in enumerate(texts):
#         file.write(str(labels[l]) + "\t" + t + "\n")


def tab_separated_ds(filepath, class_names, labelfirst=True):
    labels = []
    texts = []
    l = ''
    t = ''
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if labelfirst:
                l, t = line.strip().split('\t', 1)
            else:
                t, l = line.strip().split('\t', 1)
            if l == class_names[0]:
                labels.append(0)
            else:
                labels.append(1)
            texts.append(t)
    return labels, texts

class_names_spam = ['ham', 'spam']
class_names_sem = ['0', '1']
class_names_imdb = ['0', '1']


#                                        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DATASET = "imdb" # "spam", "sem", "imdb" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

labels = []
texts = []
if DATASET == "sem":
    l1, t1 = tab_separated_ds(SEMFILE + SEMFILE_1, class_names_sem, False)
    l2, t2 = tab_separated_ds(SEMFILE + SEMFILE_2, class_names_sem, False)
    l3, t3 = tab_separated_ds(SEMFILE + SEMFILE_3, class_names_sem, False)

    labels = l1 + l2 + l3
    texts = t1 + t2 + t3

    CLASS_NAMES = class_names_sem

elif DATASET == "spam":
    labels, texts = tab_separated_ds(SPAMFILE, class_names_spam)

    CLASS_NAMES = class_names_spam

elif DATASET == "imdb":
    labels, texts = tab_separated_ds(IMDB_COMP, class_names_imdb)

    CLASS_NAMES = class_names_imdb

# #                                                                 ||||||||||
# #                                                                 ||||||||||
# #                                                                 \/\/\/\/\/
# t_train, t_test, bert_train, bert_test, y_train, y_test = get_data(IMDB_DATA, BERT_FOLD, data=(texts, labels), text_too=True)
# #                                                                 /\/\/\/\/\
# #                                                                 ||||||||||
# #                                                                 ||||||||||

# vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
# train_vectors = vectorizer.fit_transform(t_train)
# test_vectors = vectorizer.transform(t_test)


model_params = [("rf", "i", 100), ("rf", "i", 500), ("rf", "b", 100), ("rf", "b", 500),
                ("mlp", "i", [50, 25]), ("mlp", "i", [100, 50]), ("mlp", "i", [200, 100]),
                ("mlp", "b", [50, 25]), ("mlp", "b", [100, 50]), ("mlp", "b", [200, 100])]

    

# (num_feats, num_samples, mask_method, num_rand_trees, word_level)
parameter_sets = [(5, 1000, 1, 25, True), 
                  (10, 1000, 1, 25, True), 
                  (20, 1000, 1, 25, True), 
                  (5, 1000, 1, 50, True), 
                  (10, 1000, 1, 50, True), 
                  (20, 1000, 1, 50, True), 
                  (5, 1000, 1, 100, True), 
                  (10, 1000, 1, 100, True), 
                  (20, 1000, 1, 100, True),
                  (5, 1000, 1, 200, True), 
                  (10, 1000, 1, 200, True), 
                  (20, 1000, 1, 200, True)]


descs = {
    # "models": ["RF_500_BERT", "MLP_(50-25)_BERT", "RF_500_TFIDF", "MLP_(50-25)_TFIDF"],
    "models": [model_save_name(p, DATASET, ext=False) for p in model_params],
    "parses": ["Dep", "Con", "Ran", "Std"],
    # "param_sets": ["0", "1", "2", "3"],

  #            ||||||||||||||
  #            \/\/\/\/\/\/\/
    "disting": "IMDBResults1"
} #            ^^^^^^^^^^^^^^
  #            ^^^^^^^^^^^^^^

comp_descs = {
    "models": [model_save_name(p, DATASET, ext=False) for p in model_params],
    "parses": ["Dep", "Con", "Ran"],
    "disting": "SemResults1"
}

instance_idxs = list(range(25))
# [953, 1091, 1089, 1087, 1080, 1078, 1076, 
#              1075, 1074, 1071, 1068, 1061, 1058, 1052, 1047]

# instances = [t_test[i] for i in instance_idxs]
# for i in instances:
#     print(i)


# all_models = train_models(model_params, DATASET)
# all_models = load_models(model_params, DATASET)

# run_all_explainers(all_models, CLASS_NAMES, parameter_sets, 
#                    instances, save=True, descriptions=descs, path=EXPL_PATH, skip_existing=True, just_desc=False)


def get_exp_metrics(comp_descs, all_results=False):

    loaded = load_explanations(comp_descs, EXPL_PATH, specific=True)

    # for i, ep in enumerate(loaded):
    #     print("[" + str(i) + "]: " + str(ep.get_desc()) + "\n" + ep.get_text() + "\n")
        #print(f"\nREWRITING {ep.get_path()}{ep.get_name()}\n")
        #SavedExplanation(ep.get_name(), ep.get_path(), ep.get_desc(), ep.get_exp())

    patterns = []
    for model in comp_descs["models"]:
        disting = ''
        if not all_results:
            disting = comp_descs["disting"] + "_"
        else:
            disting = ".*"
        patterns.append(re.compile(rf"^{re.escape(disting)}.*{re.escape(model)}_\d+.*"))

    sorted_exps = [[] for _ in comp_descs["models"]]
    for exp in loaded:
        for i, pattern in enumerate(patterns):
            if pattern.match(exp.get_desc()):
                sorted_exps[i].append(exp)

    for i, exp_arr in enumerate(sorted_exps):
        if len(exp_arr) > 0:
            sz = avg_influence_sz(exp_arr)
            print("\n" + comp_descs["models"][i])
            print("Average size of influence:\t" + str(sz[0]))
            print("Weighted size of influence:\t" + str(sz[1]))
            print("Avg relation distance:\t" + str(sz[2]))
            print("Weighted relation distance:\t" + str(sz[3]))


get_exp_metrics(comp_descs)

comp_descs["disting"] = "SemResults1"
get_exp_metrics(comp_descs)


# single_inst = t_test[instance_idxs[0]]
# single_inst_exps = []
# for ep in loaded:
#     print(ep.get_text() + "\t" + get_explr(ep.get_desc()) + "\n")
#     if ep.get_text().strip() == single_inst.strip():
#         single_inst_exps.append(ep)

# #exp_idxs = list(range(1,20))

# print(len(loaded))
# print(len(single_inst_exps))

# for i in range(len(single_inst_exps) - 1):
#     print(get_explr(single_inst_exps[i].get_desc()))
#     # print("Models:\t"+str(get_model(single_inst_exps[exp_idxs[i]].get_desc()))
#     #       +"\tvs.\t"+str(get_model(single_inst_exps[exp_idxs[i+1]].get_desc())))
#     print(str(exp_cos_similarity(single_inst_exps[i].all_features(), single_inst_exps[i+1].all_features()))+"\n")







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
