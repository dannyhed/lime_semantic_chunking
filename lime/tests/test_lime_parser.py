#%%
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.serialization._validate_device = lambda storage, location, backend_name=None: torch.device("cpu")

import shap.maskers
import stanza
import random as rand
import numpy as np

#import lime
import sklearn
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

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import html

import itertools as it

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from metrics import *

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
import pandas as pd
import shap
import html

EXPL_PATH = r"./saved_explanations/"
MODEL_PATH = r"./saved_models/"
HTML_PATH = r"./HTML_results/"


def clear_lines(n):
# Move up and clear n lines
    for _ in range(n):
        sys.stdout.write("\033[F")  # move cursor up
        sys.stdout.write("\033[2K") # clear line
    sys.stdout.flush()


def run_all_explainers(models, class_names, parameter_sets, instances, save=False, descriptions=None, path=None, 
                       skip_existing=False, just_desc=False, shap_train=[]):

    def save_name(j, m, i, p, desc):
        name = desc["disting"] + "_" + desc["parses"][j] + "_" + desc["models"][m] + "_" + str(p) + "_" + str(i)
        return name

    def save_desc(j, m, p, i, fts=None, smp=None, desc=None, msk=None, rnd=None, wrd=None):
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

    vectorizer = None

    for m, model in enumerate(models):
        for i, inst in enumerate(instances):
            inst = html.unescape(inst).strip()
            # Remove tokens like <#>, <unk>, or HTML tags
            inst = re.sub(r"<[^>]*>", "", inst)

            # Remove anything that isn't a word or number
            inst = ' '.join(re.findall(r"\b\w+\b", inst))
            if len(inst) == 0:
                continue
            # prediction = model([inst])
            for p, pset in enumerate(parameter_sets):
                progress = m*tot_insts*tot_params + i*tot_params + p + 1
                print(f"\n[  {int((1000 * progress) / total) / 10.0}%  |  {progress}/{total}  ]\n\n", end='', flush=True)
                (num_feats, num_samples, mask_method, num_rand_trees, word_level) = pset

                if just_desc:
                    skip_existing = True 

                new_expls = 0
    
                name = save_desc(0,m,p,i,num_feats,num_samples,descriptions,mask_method,wrd=word_level)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerDep.explain_instance(inst, model.predict_proba, num_features=num_feats, num_samples=num_samples, 
                                                                    mask_method=mask_method, word_level=word_level))
                    new_expls += 1
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    new_expls += 1

                
                name = save_desc(1,m,p,i,num_feats,num_samples,descriptions,mask_method)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerCon.explain_instance(inst, model.predict_proba, num_features=num_feats, num_samples=num_samples, 
                                                                    mask_method=mask_method))
                    new_expls += 1
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    new_expls += 1
                    
                name = save_desc(2,m,p,i,num_feats,num_samples,descriptions,mask_method,rnd=num_rand_trees)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerRan.explain_instance(inst, model.predict_proba, num_features=num_feats, num_samples=num_samples, 
                                                                    random_trees=num_rand_trees, mask_method=mask_method))
                    new_expls += 1
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    new_expls += 1
                    
                name = save_desc(3,m,p,i,num_feats,num_samples,descriptions)[0]
                if not (skip_existing and os.path.exists(path+name+".pkl")):
                    print("\n" + name)
                    explanations.append(explainerStd.explain_instance(inst, model.predict_proba, num_features=num_feats, num_samples=num_samples))
                    new_expls += 1
                elif just_desc:
                    explanations.append(SavedExplanation(name, path).get_exp())
                    new_expls += 1
                    
                # if len(shap_train) > 0:
                #     name = save_desc(4,m,p,i,num_feats,num_samples,descriptions)[0]
                #     if not (skip_existing and os.path.exists(path+name+".pkl")):
                #         print("\n" + name)
                #         try:
                #             vectorizer = model.named_steps["tfidfvectorizer"]
                #         except:
                #             print(model.named_steps)
                #             vectorizer = model.named_steps["bertvectorizer"].tokenizer
                #         # teacher_forcing_model = shap.models.TeacherForcing(
                #         #     model.predict_proba, tokenizer=tokenizer)
                #             #model, similarity_model=model, similarity_tokenizer=tokenizer, device=tokenizer.device)
                #         # mask = shap.maskers.Text(tokenizer)
                #         #background = vectorizer(shap_train[:100]).toarray()
                #         #print(f"Instance: {inst}")
                #         # background = shap_train[:100]
                #         # mask = shap.maskers.Text(r"\W+")
                #         # sh = shap.KernelExplainer(model.predict_proba, background)
                #         smodel = shap.models.TransformersPipeline(model.predict_proba, rescale_to_logits=False)
                #         sh = shap.Explainer(smodel)
                #         explanations.append(sh(inst))
                #         new_expls += 1
                #     elif just_desc:
                #         explanations.append(SavedExplanation(name, path).get_exp())
                #         new_expls += 1

                if just_desc:
                    skip_existing = False

                if save:    
                    name, desc = save_desc(0,m,p,i,num_feats,num_samples,descriptions,mask_method,wrd=word_level)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-new_expls])
                        new_expls -= 1
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(1,m,p,i,num_feats,num_samples,descriptions,mask_method)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-new_expls])
                        new_expls -= 1
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(2,m,p,i,num_feats,num_samples,descriptions,mask_method,rnd=num_rand_trees)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-new_expls])
                        new_expls -= 1
                    else:
                        print("\n" + name + " exists" + "\n")
                    name, desc = save_desc(3,m,p,i,num_feats,num_samples,descriptions)
                    if not(skip_existing and os.path.exists(path+name+".pkl")):
                        SavedExplanation(name, path, desc, explanations[-new_expls])
                        new_expls -= 1
                    else:
                        print("\n" + name + " exists" + "\n")
                    # name, desc = save_desc(4,m,p,i,desc=descriptions)
                    # if not(skip_existing and os.path.exists(path+name+".pkl")):
                    #     SavedExplanation(name, path, desc, explanations[-new_expls], 
                    #                      addl_data={"predict_proba": model.predict_proba(inst)})
                    #     new_expls -= 1
                    # else:
                    #     print("\n" + name + " exists" + "\n")
                        
                
                clear_lines(15)

    return explanations

def load_explanations(descs, path, specific=False):
    
    explanations = []

    if specific:
        patterns = []

        for parse in descs["parses"]:
            for model in descs["models"]:
                disting = descs["disting"]
                patterns.append(re.compile(rf"^{re.escape(disting)}_{re.escape(parse)}_{re.escape(model)}.*\.pkl$"))
        
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
    return my_cos_similarity(exp1, exp2)
    
def my_cos_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    try:
        return dot / (norm1 * norm2)
    except:
        return 0

from pythainlp.tokenize import word_tokenize

class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', language="En", device=None):
        if language == "Leb":
            model_name = "aubmindlab/bert-base-arabert"
        elif language == "Urdu":
            model_name = "callmesan/ModernBERT-large-roman-urdu-binary"
        elif language == "Thai":
            model_name = "monsoon-nlp/bert-base-thai"
        elif language == "Turk":
            model_name = "dbmdz/bert-base-turkish-cased"
        elif language == "Beng":
            model_name = "csebuetnlp/banglabert"
        self.model_name = model_name
        self.lang = language
        self.device = device or 'cpu' # ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        vectors = []
        for sentence in X:
            if self.lang == "Thai":
                sentence = " ".join(word_tokenize(sentence, engine="newmm"))

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
        texts, labels, lang = dataset
    except:
        print("ERROR: No data provided!")
        return None
    bert_sens = []
    save_data = []
    bert_vectorizer = BERTVectorizer(language=lang)
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
        desc = exp.get_desc()
        m = desc.split('\n')[1].split('\t')[2]
        i = desc.split('\n')[0].split('_')[-1]
        print(m + "\t" + i)
    # exp.get_exp().save_to_file("INFLUENCETEST.html")

    if not exp.data["is_standard"]:
        local_exp = exp.get_local_exp()
        if len(local_exp) == 0:
            print(exp.get_tokens())
            print(exp.get_text())
            print(exp.get_idx_string().raw_string)
        ids = [x[0] for x in local_exp]

        dependence_chunks = [[] for _ in range(int(max(ids)/num_words) + 1)]
        id_chunks = [[] for _ in range(int(max(ids)/num_words))]
        weighted_dep_chunks = [0 for _ in range(int(max(ids)/num_words))]
        for i in local_exp:
            dependence_chunks[int(i[0]/num_words)].append(i)
        for i, chunk in enumerate(dependence_chunks[1:]):
            try:
                weighted_dep_chunks[i] = chunk[0][1]
            except:
                return (0, 0, 0, 0, 0)
                # exp.get_exp().save_to_file("problematic.html")
                # print(exp.get_desc())
                # print(num_words)
                # print(local_exp)
                # print(dependence_chunks)
                # print(chunk)
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
        if len(exp.get_local_exp()) > 0:
            infl_sz = influence_sz(exp)
            # print(infl_sz)
            if infl_sz[4] != 0:
                avg_sz += infl_sz[0]
                wgh_sz += infl_sz[1]
                dist_avg += infl_sz[2]
                wgh_dst += infl_sz[3]
                word_sum += infl_sz[4]
    if word_sum != 0:
        avg_sz = avg_sz / word_sum
        wgh_sz = wgh_sz / word_sum
        dist_avg = dist_avg / word_sum
        wgh_dst = wgh_dst / word_sum
        return (avg_sz, wgh_sz, dist_avg, wgh_dst)
    else:
        return (0, 0, 0, 0)


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


def train_models(all_model_params, train_vectors, bert_train, y_train, vectorizer, 
                 dataset=None, language="En", jmodel=False, skip_existing=True, pretrained_clf=None):
    print("Training Models...")
    models_trained = []
    bert_vectorizer = BERTVectorizer(language=language)
    for p in tqdm(all_model_params):
        if not (skip_existing and os.path.exists(MODEL_PATH + model_save_name(p, dataset))):
            m = []
            if p[0] == "rf":
                m = sklearn.ensemble.RandomForestClassifier(n_estimators=p[2])
            elif p[0] == "mlp":
                m = MLPClassifier(solver="lbfgs", alpha=1e-5,
                                hidden_layer_sizes=tuple(p[2]), random_state=1, max_iter=2000)
            if p[1] == "i":
                m.fit(train_vectors, y_train)
                if jmodel:
                    models_trained.append(make_pipeline(vectorizer, m))
                else:
                    models_trained.append(make_pipeline(vectorizer, m).predict_proba)
            elif p[1] == "b":
                m.fit(bert_train, y_train)
                if jmodel:
                    models_trained.append(make_pipeline(bert_vectorizer, m))
                else:
                    models_trained.append(make_pipeline(bert_vectorizer, m).predict_proba)
            with open(MODEL_PATH + model_save_name(p, dataset), "wb") as file:
                pkl.dump(models_trained[-1], file)
            print(f"model saved to {MODEL_PATH + model_save_name(p, dataset)}")
        else:
            print(f"{MODEL_PATH + model_save_name(p, dataset)} exists...")
    clear_lines(1)
    print("Models trained")
    return models_trained

def load_models(all_model_params, dataset=None, return_name=False, reload_vec=False):
    print("Loading models...")
    models_loaded = []
    for p in tqdm(all_model_params, dataset):
        name = model_save_name(p, dataset)
        with open(MODEL_PATH + name, "rb") as file:
            if return_name:
                models_loaded.append((pkl.load(file), name))
            else:
                models_loaded.append(pkl.load(file))
        if reload_vec:
            old_pipe = models_loaded[-1]
            ( o_v, same_clf) = (old_pipe.steps[0][1], old_pipe.steps[1][1])
            # print(f"TYPE OF o_v: {type(o_v)}\tTYPE OF same_clf: {type(same_clf)}")
            # if isinstance(o_v, BERTVectorizer):
            new_vec = BERTVectorizer(model_name=o_v.model_name, language=o_v.lang)
            # else:
            #     print("Please fix load_models to accomadate for other vectorizers")
            models_loaded[-1] = make_pipeline(new_vec, same_clf)

            
    # clear_lines(1)
    print("Models loaded")
    return models_loaded

def print_explanations(disting, names=None, more_name=None, add_regex=None):
    if add_regex == None:
        pattern = re.compile(f"^{re.escape(disting)}.*\.pkl$")
    else:
        pattern = re.compile(f"^{re.escape(disting)}{add_regex}\.pkl$")
    print(pattern)
    files = []
    for file in os.listdir(EXPL_PATH):
        if pattern.match(file):
            print("found file")
            files.append(file)
    files = sorted(files)
    if names == None:
        for f, file in enumerate(files):
            pathname = HTML_PATH + disting + more_name + str(f)
            exp = SavedExplanation(file, EXPL_PATH).get_exp()
            exp.save_to_file(pathname + ".html")
            print(pathname)
            with open(pathname + ".txt", "w+") as file:
                file.write(str(exp.predict_proba[0]))

    else:
        for name, file in zip(names, files):
            pathname = HTML_PATH + disting + more_name + name
            exp = SavedExplanation(file, EXPL_PATH).get_exp()
            exp.save_to_file(pathname + ".html")
            print(pathname)
            with open(pathname + ".txt", "w+") as file:
                file.write(str(exp.predict_proba[0]))




DATASETS = r"./datasets/"
BERT_FOLD = r"./bert_data/"
SPAM_DATA = "spam_ds"
SEM_DATA = "sem_ds"
IMDB_DATA = "imdb_ds"
HATE_DATA = "hate_ds"
SPAMFILE = os.path.join(DATASETS, "smsspamcollection/SMSSpamCollection")
SEMFILE = os.path.join(DATASETS, "sentiment_sens/")
SEMFILE_1 = "amazon_cells_labelled.txt"
SEMFILE_2 = "imdb_labelled.txt"
SEMFILE_3 = "yelp_labelled.txt"
IMDB = os.path.join(DATASETS, "aclImdb/")
IMDB_TT = ["test/", "train/"]
IMDB_NP = ["neg/", "pos/"]
IMDB_COMP = IMDB + "imdb_compiled.txt"
HATEFILE = os.path.join(DATASETS, "hate_speech/labeled_data.csv")
HATETAB = os.path.join(DATASETS, "hate_speech/tab_sep_hate_data.csv")

LEB_AR_REVS = os.path.join(DATASETS, "Lebanese_Arabic_Reviews/Lebanese_Arabic_Reviews.csv")
ROM_URDU_SENT = os.path.join(DATASETS, "Roman_Urdu_Sentiment/urdu_sents.tsv")
BENG_HATE_FOLD = os.path.join(DATASETS, "Bengali_Hate_Speech/")
BENG_TRAIN = os.path.join(BENG_HATE_FOLD, "train.csv")
BENG_TEST = os.path.join(BENG_HATE_FOLD, "test.csv")
BENG_VAL = os.path.join(BENG_HATE_FOLD, "validate.csv")
TURK_SPAM = os.path.join(DATASETS, "Turkish_Spam/trspam.csv")
THAI_SENT_FOLD = os.path.join(DATASETS, "Wisesight_Thai_Sentiment/")
THAI_NEG = os.path.join(THAI_SENT_FOLD, "neg.txt")
THAI_NEU = os.path.join(THAI_SENT_FOLD, "neu.txt")
THAI_POS = os.path.join(THAI_SENT_FOLD, "pos.txt")
THAI_Q = os.path.join(THAI_SENT_FOLD, "q.txt")

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

# df = pd.read_parquet("hf://datasets/ucberkeley-dlab/measuring-hate-speech/measuring-hate-speech.parquet")
# with open(r"./hate_speech/measuring_hate.csv", "wb") as file:
#     df.to_csv(file)

# texts = []
# labels = []
# with open(HATEFILE, "r", encoding='utf-8') as file:
#     file.readline()
#     secs = []
#     label = 0
#     for i, line in enumerate(file):
#         secs = line.split(",")
#         try:
#             label = secs[5]
#         except:
#             continue
#         t = secs[6]
#         t = t.replace("\"", "")
#         t = re.sub(r"^.*?@.*?:", "", t, count=1)
#         t = re.sub("&.*?;", "", t)
#         t = re.sub("@.*? ", "", t)
#         t = re.sub("https*://.*?[ \n]", "", t)
#         t = t.replace("  ", " ")
#         t = t.strip()
#         if t != '':
#             texts.append(t)
#             if label == '0' or label == '1':
#                 labels.append(1)
#             else:
#                 labels.append(0)

# with open(HATETAB, "w+") as file:
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

class_names_spam = ['0', '1'] #['ham', 'spam']
class_names_sem = ['0', '1']
class_names_imdb = ['0', '1']
class_names_hate = ['0', '1']


def get_ts_and_ls(DS):
    labels = []
    texts = []
    if DS == "sem":
        l1, t1 = tab_separated_ds(SEMFILE + SEMFILE_1, class_names_sem, False)
        l2, t2 = tab_separated_ds(SEMFILE + SEMFILE_2, class_names_sem, False)
        l3, t3 = tab_separated_ds(SEMFILE + SEMFILE_3, class_names_sem, False)

        labels = l1 + l2 + l3
        texts = t1 + t2 + t3

        CLASS_NAMES = class_names_sem

        lang = "en"

    elif DS == "hate_beng":
        train = pd.read_csv(BENG_TRAIN, header=0)
        test = pd.read_csv(BENG_TEST, header=0)
        val = pd.read_csv(BENG_VAL, header=0)

        texts = [re.sub(r'\{Emoji}', '', (html.unescape(x)).strip()).replace('\"', '') for x in train[train.columns[0]]]
        texts.extend([re.sub(r'\{Emoji}', '', (html.unescape(x)).strip()).replace('\"', '') for x in test[test.columns[0]]])
        texts.extend([re.sub(r'\{Emoji}', '', (html.unescape(x)).strip()).replace('\"', '') for x in val[val.columns[0]]])

        labels = [0 if x == "Neutral" else 1 for x in train[train.columns[1]]]
        labels.extend([0 if x == "Neutral" else 1 for x in test[train.columns[1]]])
        labels.extend([0 if x == "Neutral" else 1 for x in val[val.columns[1]]])

        lang = "bn"


    elif DS == "spam_turk":
        data = pd.read_csv(TURK_SPAM, header=0, quotechar='"', on_bad_lines="skip")
        labels = [1 if x == "spam" else 0 for x, y in zip(data[data.columns[1]], data[data.columns[0]]) if pd.notnull(y) and pd.notnull(x)]
        texts = [re.sub(r'\{Emoji}', '', (html.unescape(x)).strip()).replace('\"', '') for x, y in zip(data[data.columns[0]], data[data.columns[1]]) if pd.notnull(y) and pd.notnull(x)]

        lang = "tr"

    elif DS == "sent_thai":
        with open(THAI_NEG, "r", encoding="utf-8") as file:
            lines = file.readlines()
        labels, texts = [0 for line in lines if re.sub(r'\{Emoji}', '', line.strip()).strip() != ''], [re.sub(r'\{Emoji}', '', line.strip()) for line in lines if re.sub(r'\{Emoji}', '', line.strip()).strip() != '']

        with open(THAI_POS, "r", encoding="utf-8") as file:
            lines = file.readlines()
        labels.extend([1 for line in lines if re.sub(r'\{Emoji}', '', line.strip()).strip() != ''])
        texts.extend([re.sub(r'\{Emoji}', '', line.strip()) for line in lines if re.sub(r'\{Emoji}', '', line.strip()).strip() != ''])

        lang = "th"

    elif DS == "sent_urdu":
        with open(ROM_URDU_SENT, encoding="utf-8") as f:
            lines = f.readlines()

        labels, texts = [], []

        for line in lines:
            line = html.unescape(line.replace('"', '').strip())
            if '\t' in line:
                label, text = line.split('\t', 1)
                labels.append(label.strip())
                texts.append(text.strip())

        lang = "ur"
    
    elif DS == "sent_leb":
        data = pd.read_csv(LEB_AR_REVS, header=0)
        labels, texts = ([1 if x >=3 else 0 for x in data[data.columns[2]]], 
                        [re.sub(r'\{Emoji}', '', x.strip()).replace("\"", "") for x in data[data.columns[1]]])
        
        lang = "ar"
            
    elif DS == "spam":
        labels, texts = tab_separated_ds(SPAMFILE, class_names_spam)

        CLASS_NAMES = class_names_spam

        lang = "en"

    elif DS == "imdb":
        labels, texts = tab_separated_ds(IMDB_COMP, class_names_imdb)

        CLASS_NAMES = class_names_imdb

        lang = "en"

    elif DS == "hate":    
        labels, texts = tab_separated_ds(HATETAB, class_names_hate)

        CLASS_NAMES = class_names_hate

        lang = "en"
    
    return texts, labels, lang



comp_descs = {}



instance_idxs = list(range(25))
params = [6]
models_1 = [-1]
dists_1 = ["SpamHuman1", "SemHuman1", "IMDBHuman1",
            "HateHuman1"]
# run_all_datasets(dists_1, models_1, params, instance_idxs)


instance_idxs = list(range(25, 50))
models_2 = [0]
# "spam", "sem", "imdb", "hate"
dists_2 = ["SpamHuman2", "SemHuman2", 
            "IMDBHuman2", "HateHuman2"]
#"SpamResults1", "SemResults1", "IMDBResults1",
            # "HateResults1", 

# run_all_datasets(dists_2, models_2, params, instance_idxs)


all_dists = ["SpamHuman1", "SemHuman1", "IMDBHuman1",
            "HateHuman1", "SpamHuman2", "SemHuman2", 
            "IMDBHuman2", "HateHuman2"]
models = [0, -1]

# comp_descs = {
#     "models": [model_save_name(MODEL_PARAMS[p], dataset=None, ext=False) for p in models],
#     "parses": ["Dep", "Con", "Ran", "Std"],
#     "params": [EXP_PARAMS[p] for p in params],
#     "instances": instance_idxs,
#     "disting": "SpamResults1"
# }

def get_exp_metrics(comp_descs, compare_by="model", all_results=False):

    loaded = load_explanations(comp_descs, EXPL_PATH, specific=False) 
    print(f"Found {len(loaded)} explanations...")

    # for i, ep in enumerate(loaded):
    #     print("[" + str(i) + "]: " + str(ep.get_desc()) + "\n" + ep.get_text() + "\n")
        #print(f"\nREWRITING {ep.get_path()}{ep.get_name()}\n")
        #SavedExplanation(ep.get_name(), ep.get_path(), ep.get_desc(), ep.get_exp())

    patterns = []
    sorted_exps = []

    disting = ''
    if not all_results:
        disting = re.escape(comp_descs["disting"])
    else:
        disting = ".*"

    if compare_by == "model":
        for model in comp_descs["models"]:
            patterns.append(re.compile(f"^{disting}_.*{re.escape(model)}_\d+.*\n"))
        sorted_exps = [(m, []) for m in comp_descs["models"]]
    
    elif compare_by == "exp_params":
        for p, _ in enumerate(comp_descs["params"]):
            patterns.append(re.compile(f"^{disting}_.*{re.escape(p)}_\d+\n"))
        sorted_exps = [(str(i), []) for i in range(len(comp_descs["params"]))]

    elif compare_by == "parse":
        for p in comp_descs["parses"]:
            patterns.append(re.compile(f"^{disting}_{re.escape(p)}_.*\n"))
        sorted_exps = [(p, []) for p in comp_descs["parses"]]

    elif compare_by == "inst":
        for i in comp_descs["instances"]:
            patterns.append(re.compile(f"^{disting}.*_{i}\n"))
        sorted_exps = [(str(i), []) for i in comp_descs["instances"]]

    elif compare_by == "exp":
        for parse in comp_descs["parses"]:
            for pars, _ in enumerate(comp_descs["params"]):
                patterns.append(re.compile(f"^{disting}_{re.escape(parse)}_.*_{pars}_\d+\n"))
        sorted_exps = [(f'{comp_descs["parses"][int(i/len(comp_descs["params"]))]}-{i%len(comp_descs["params"])}', []) for i in range(len(comp_descs["parses"]) * len(comp_descs["params"]))]

    print(f"{len(patterns)} patterns found...")

    found = [0 for _ in sorted_exps]

    for exp in loaded:
        desc = exp.get_desc()
        for i, pattern in enumerate(patterns):
            if pattern.match(desc):
                found[i] += 1
                sorted_exps[i][1].append(exp)
                continue

    for i, f in enumerate(found):
        print(f"{patterns[i]}: {f}")

    print(f"{sum([len(i) for (_, i) in sorted_exps])} explanations total...")

    for i, (name, exp_arr) in enumerate(sorted_exps):
        if len(exp_arr) > 0:
            sz = avg_influence_sz(exp_arr)
            print("\n" + name)
            print("Average size of influence:\t" + str(sz[0]))
            print("Weighted size of influence:\t" + str(sz[1]))
            print("Avg relation distance:\t\t" + str(sz[2]))
            print("Weighted relation distance:\t" + str(sz[3]))

def get_all_distings():
    all_dists = set()
    for file in os.listdir(EXPL_PATH):
        all_dists.add(SavedExplanation(file, EXPL_PATH).get_desc().split("\n")[0].split("_")[0])

    for dist in all_dists:
        print(dist)
    return all_dists


# ALL_DATASETS = ["spam", "sem", "imdb", "hate", "spam", "sem", "imdb", "hate"] 

# all_dists = ["SpamResults1", "SemResults1", "IMDBResults1",
#             "HateResults1", "SpamResults2", "SemResults2", 
#             "IMDBResults2", "HateResults2"]

# more_name = ''
# models = comp_descs["models"]
# for dist, ds in zip(all_dists, ALL_DATASETS):
#     for parse in comp_descs["parses"]:
#         for m in models:
#             if m == models[0]:
#                 more_name = "small"
#             else:
#                 more_name = "large"
#             print(more_name)
#             print_explanations(dist, add_regex=f"_{parse}_{ds}-{re.escape(m)}_{params[0]}_\d+", more_name=(parse + "_" + more_name))

par = 0

# more_name = "large"
# m = model_save_name(MODEL_PARAMS[models_1[0]], ext=False)
# for dist, ds in zip(dists_1, ALL_DATASETS):
#     for parse in comp_descs["parses"]:
#         print_explanations(dist, add_regex=f"_{parse}_{ds}-{re.escape(m)}_{par}_\d+", more_name=(parse + "_" + more_name))

# more_name = "small"
# m = model_save_name(MODEL_PARAMS[models_2[0]], ext=False)
# for dist, ds in zip(dists_2, ALL_DATASETS):
#     for parse in comp_descs["parses"]:
#         print_explanations(dist, add_regex=f"_{parse}_{ds}-{re.escape(m)}_{par}_\d+", more_name=(parse + "_" + more_name))


# get_exp_metrics(comp_descs, compare_by="exp")

# # run_all_datasets(all_dists=ALL_DATASETS, model_param_sets=[0, 1], exp_param_sets=None, instance_idxs=None)



# ALL_DATASETS = ["sent_leb", "sent_urdu", "sent_thai", 
#                 #"spam", 
#                 "spam_turk", 
#                 #"hate", 
#                 "hate_beng"]

import unicodedata

def normalize_text(text):
    # Normalize to NFC and remove zero-width characters
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # zero-width non-joiner/joiner
    return text.strip()

def explain_multilang(instances, models, language="ar", filenames_dict={}, path=HTML_PATH, samples=100, class_names=[0, 1]):
    # filenames_dict = {"model_names": ["mlp_large", "mlp_small"],
    #                   "begin_filename": "ar",                         
    #                   "post_name": "test", "first_try", "final", etc...,   
    #                   "parse_names": ["dep", "ran", "std"],                
    #                   "inst_names": ["confusing1", "confusing2", "simple1"]}

    if not isinstance(instances, list):
        instances = [instances]
    if not isinstance(models, list):
        models = [models]
    if "model_names" not in filenames_dict:
        filenames_dict["model_names"] = [f"model{x}" for x in range(len(models))]
    if "inst_names" not in filenames_dict:
        filenames_dict["inst_names"] = [str(x) for x in range(len(instances))]
    if "parse_names" not in filenames_dict:
        filenames_dict["parse_names"] = ["dep", "ran", "std"]
    if "post_name" not in filenames_dict:
        filenames_dict["post_name"] = "exp"
    if "begin_filename" not in filenames_dict:
        filenames_dict["begin_filename"] = language + "_"
    elif filenames_dict["begin_filename"] != "":
        filenames_dict["begin_filename"] = filenames_dict["begin_filename"] + "_"
    
    fld = filenames_dict

    combo_names = it.product(fld["model_names"], fld["inst_names"], fld["parse_names"])
    combo_exps = it.product(models, instances, fld["parse_names"])

    filenames = [f"{fld['begin_filename']}{parse}_{model}_{fld['post_name']}_{inst}.html" for (model, inst, parse) in combo_names]

    explainers = {"std": LimeTextExplainer(class_names=class_names, verbose=False),
                  "ran": LimeTextParserExplainer(class_names=class_names, verbose=False, language=language, parsing_type="random"),
                  "dep": LimeTextParserExplainer(class_names=class_names, verbose=False, language=language, parsing_type="dependency")}
    if language == "en":
        explainers["con"] = LimeTextParserExplainer(class_names=class_names, verbose=False, language=language, parsing_type="constituency")

    for exp, fname in tqdm(list(zip(combo_exps, filenames)), "Explaining"):
        m, i, p = exp
        # print(i)
        i = normalize_text(i)
        with open(path + fname, "w+", encoding="utf-8") as file:
            file.write(explainers[p].explain_instance(i, m, num_samples=samples).as_html())
import spacy
import random

nlp = spacy.load("en_core_web_md")

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def extract_influences(exp, label):
    """
    Converts LIME local_exp to a fixed-length float vector.
    """
    SavedExplanation("temp_sexp", EXPL_PATH, "None", exp)
    saved_exp = SavedExplanation("temp_sexp", EXPL_PATH, "None")
    all_feats = saved_exp.all_features(label)

    vector = [0] * (max([feat[0] for feat in all_feats]) + 1)

    for feat in all_feats:
        vector[feat[0]] = feat[1]

    return vector
    # local_exp = exp.local_exp[label]

    # exp_ids = [x[0] for x in local_exp]
    # tokens = list(self.get_tokens().keys())
    # for i in range(len(tokens)):
    #     if tokens[i] in exp_ids:
    #         j = 0
    #         while exp_ids[j] != tokens[i]:
    #             j += 1
    #         complete_exp.append(local_exp[j])
    #     else:
    #         complete_exp.append((tokens[i], 0.0))

    # influences = np.zeros(num_features, dtype=np.float32)

    # for idx, weight in exp.local_exp[label]:
    #     if idx < num_features:
    #         influences[idx] = weight

    # return influences


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def compute_lime_stability(
    explanations,
    distance_metric=euclidean_distance,
    metric="similarity"
):
    """
    Compute LIME stability (m_f9.2) per dataset.

    Parameters
    ----------
    explanations : np.ndarray
        Shape: (datasets, sentences, neighbors, features)
    distance_metric : function
        Distance function between explanation vectors
    metric : str
        "similarity" or "identity"

    Returns
    -------
    stability_scores : np.ndarray
        Shape: (datasets,)
    """

    stability_scores = np.zeros(len(explanations), dtype=np.float32)

    for d in range(len(explanations)):
        # For dataset d, collect neighbors for each sentence
        exp_neighbors = [
            explanations[d, i]
            for i in range(len(explanations[d]))
        ]

        stability_scores[d] = f9_score(
            exp_neighbors=exp_neighbors,
            distance_metric=distance_metric,
            metric=metric
        )

    return stability_scores

def get_similar_word(token, top_n=20, similarity_threshold=0.55):
    """Find a similar word with same POS using vectors."""
    if not token.has_vector:
        return token.text

    candidates = []
    for word in nlp.vocab:
        if (
            word.is_lower
            and word.has_vector
            and word.prob >= -15
            and nlp(word.text)[0].pos_ == token.pos_
        ):
            sim = token.similarity(nlp(word.text)[0])
            if sim >= similarity_threshold and word.text != token.text:
                candidates.append(word.text)

    return random.choice(candidates) if candidates else token.text


def synonym_swap(sentence, replace_prob=0.3):
    doc = nlp(str(sentence))
    new_tokens = []

    for token in doc:
        if (
            token.pos_ in ALLOWED_POS
            and not token.is_stop
            and random.random() < replace_prob
        ):
            new_tokens.append(get_similar_word(token))
        else:
            new_tokens.append(token.text)

    new_sentence = spacy.tokens.Doc(doc.vocab, words=new_tokens).text

    for t1, t2 in zip(doc, new_tokens):
        if t1.text != t2:
            print(f"Old sentence: {sentence}\nNew sentence: {new_sentence}")

    return new_sentence

def generate_syns(sentences, replace_prob=0.3):
    return [synonym_swap(sen, replace_prob) for sen in sentences]



#["imdb", 
ALL_DATASETS = ["imdb", "sem"]
                #"sent_leb"], 
                #"sent_urdu", "sent_thai"] 
                #"spam", "spam_turk", 
                #"hate", "hate_beng"] # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dss = {}

for ds in ALL_DATASETS:
    data = get_ts_and_ls(ds)
    dss[ds] = get_data(ds, BERT_FOLD, data=data, text_too=True)



#[("rf", "i", 100), ("rf", "i", 500), ("rf", "b", 100), ("rf", "b", 500),
                #("mlp", "i", [50, 25]), ("mlp", "i", [100, 50]), ("mlp", "i", [200, 100]),
MODEL_PARAMS = [("mlp", "b", [50, 25]), #("mlp", "b", [100, 50]), 
                ("mlp", "b", [200, 100])]


def similar_explanations(
    explainer,
    models,
    dataset,
    num_sens=50,
    num_syns=10):

    # Load models and sample sentences
    sample_sens = np.random.choice(dss[dataset][1], num_sens, replace=False)

    # Shape: (datasets, sentences, synonyms + original)
    similar_sens = np.empty(
        (num_sens, num_syns + 1),
        dtype=object
    )

    # Insert originals
    similar_sens[:, 0] = sample_sens

    # Generate synonyms
    for i, sentence in enumerate(sample_sens):
        similar_sens[i, 1:] = generate_syns([sentence] * num_syns)

    # Allocate explanation tensor
    # explanations = np.zeros(
    #     (
    #         num_sens,
    #         num_syns + 1
    #     ),
    #     dtype=np.float32
    # )

    exp_tensor = []

    # Generate LIME explanations
    for m in models:

        for i in range(num_sens):

            model_row = []

            for j in range(num_syns + 1):

                syn_row = []

                sentence = similar_sens[i, j]

                print(f"sentence: {sentence}")

                exp = explainer.explain_instance(sentence, m.predict_proba, num_features=7)

                label = exp.available_labels()[0]

                # explanations[i, j] = 
                syn_row.append(extract_influences(exp, label))

            model_row.append(syn_row)

        exp_tensor.append(model_row)

    return exp_tensor



def stability(explainers, models, dataset):
    for key, value in explainers.items():

        explanations = similar_explanations(explainer=explainers[key], models=models, 
                                            dataset=dataset, num_sens=50, num_syns=10)

        # Compute stability
        lime_stability = compute_lime_stability(
            explanations,
            distance_metric=euclidean_distance,
            metric="similarity"
        )

        print(f"LIME stability for {dataset} per model: {lime_stability}")
    
    


def obj_metrics(num_sens=50, num_syns=10):    
    all_models = [load_models(MODEL_PARAMS, ds, reload_vec=True) for ds in ALL_DATASETS]
    explainers = {"std": LimeTextExplainer(verbose=False),
                  "ran": LimeTextParserExplainer(verbose=False, parsing_type="random"),
                  "dep": LimeTextParserExplainer(verbose=False, parsing_type="dependency"),
                  "con" : LimeTextParserExplainer(verbose=False, parsing_type="constituency")}
    
    stab = [[stability(explainers, m, ALL_DATASETS[d])] for d, m in enumerate(all_models)]
    print(stab)
    

    # all_models = []
    # sample_sens = []

    # for ds in ALL_DATASETS:
    #     all_models.append(load_models(MODEL_PARAMS, ds))
    #     sample_sens.append(np.random.choice(dss[ds][1], num_sens))

    # similar_sens = np.empty_like((len(ALL_DATASETS), num_sens, num_syns + 1), dtype=np.str_)
    # similar_sens[:][:][0] = sample_sens

    # for d, ds in enumerate(ALL_DATASETS):
    #     for i, sample in enumerate(sample_sens[d]):
    #         similar_sens[d][i][1:] = generate_syns([sample[i]]*num_syns)


        
    #     all_models[d]




    #generate explanations
    #pass to metrics.py f_9


def run_all_datasets(all_dists, model_param_sets, exp_param_sets, instance_idxs):
    model_params = [MODEL_PARAMS[i] for i in model_param_sets]
    parameter_sets = [EXP_PARAMS[i] for i in exp_param_sets]

    for dist, DS in enumerate(ALL_DATASETS):
        print(f"NEXT DATASET: {DS}")
        descs = {
            # "models": ["RF_500_BERT", "MLP_(50-25)_BERT", "RF_500_TFIDF", "MLP_(50-25)_TFIDF"],
            "models": [model_save_name(p, DS, ext=False) for p in model_params],
            "parses": ["Dep", "Con", "Ran", "Std", "Shap"],
            # "param_sets": ["0", "1", "2", "3"],

        #            ||||||||||||||
        #            \/\/\/\/\/\/\/
            "disting": all_dists[dist]
        } #            ^^^^^^^^^^^^^^
        #            ^^^^^^^^^^^^^^
        #                                                                 \/\/\/\/\/
        texts, labels, lang = get_ts_and_ls(DS)
        t_train, t_test, bert_train, bert_test, y_train, y_test = get_data(DS, BERT_FOLD, (texts, labels), text_too=True)
        #                                                                 /\/\/\/\/\

        # vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
        # train_vectors = vectorizer.fit_transform(t_train)
        # test_vectors = vectorizer.transform(t_test)
        train_vectors = None
        vectorizer = None

        # [953, 1091, 1089, 1087, 1080, 1078, 1076, 
        #              1075, 1074, 1071, 1068, 1061, 1058, 1052, 1047]

        instances = [t_test[i] for i in instance_idxs]
        # for i in instances:
        #     print(i)
        t_train = np.array(list(t_train))
        t_test = np.array(list(t_test))
        all_models = train_models(model_params, train_vectors, bert_train, y_train, vectorizer, DS, language=lang, jmodel=True)

        all_models = load_models(model_params, DS)

        run_all_explainers(all_models, [0, 1], parameter_sets, 
                        instances, save=True, descriptions=descs, path=EXPL_PATH, 
                        skip_existing=True, just_desc=False)


# (num_feats, num_samples, mask_method, num_rand_trees, word_level)
EXP_PARAMS = [(5, 1000, 1, 25, True), 
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


# bn_models = load_models([("mlp", "b", [50, 25]), ("mlp", "b", [200, 100])], dataset=ALL_DATASETS[4])


def demo_explanations():
    th_models = load_models(MODEL_PARAMS, dataset=ALL_DATASETS[4])

    tr_models = load_models(MODEL_PARAMS, dataset=ALL_DATASETS[6])

    ur_models = load_models(MODEL_PARAMS, dataset=ALL_DATASETS[3])

    ar_models = load_models(MODEL_PARAMS, dataset=ALL_DATASETS[2])

    # explain_multilang(dss["hate_beng"][0][1], [m.predict_proba for m in bn_models], "bn", {"post_name": "newtest"})
    explain_multilang(list(dss["sent_leb"][0][1:5]), [m.predict_proba for m in ar_models], "ar", {"post_name": "mdemo"})
    explain_multilang(list(dss["sent_urdu"][0][1:5]), [m.predict_proba for m in ur_models], "ur", {"post_name": "mdemo"})
    explain_multilang(list(dss["sent_thai"][0][1:5]), [m.predict_proba for m in th_models], "th", {"post_name": "mdemo"})
    explain_multilang(list(dss["spam_turk"][0][1:5]), [m.predict_proba for m in tr_models], "tr", {"post_name": "mdemo"})
# explainerDep_bn.explain_instance(dss["hate_beng"][20], bn_models_lg.predict_proba).as_html(HTML_PATH)
# explainerRan_th.explain_instance(dss["sent_thai"][20], th_models_lg.predict_proba).as_html(HTML_PATH)
# explainerRan_tr.explain_instance(dss["spam_turk"][20], tr_models_lg.predict_proba).as_html(HTML_PATH)
# explainerStd.explain_instance(dss["sent_urdu"][20], ur_models_lg.predict_proba).as_html(HTML_PATH)

def demo_models():
    combos = [(load_models(MODEL_PARAMS, dataset=d, return_name=True), d) for d in ALL_DATASETS]
    for c in combos:
        ms, d = c
        train_sens, test_sens, train_bert, test_bert, y_train, y_test = dss[d]
        for (m, n) in ms:
            y_pred = m.predict(test_sens)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"{n} test results:")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")

#return (train_sens, test_sens, train_bert, test_bert, y_train, y_test)

# demo_models()
# demo_explanations()
obj_metrics()


# comp_descs["disting"] = "SemResults1"
# get_exp_metrics(comp_descs)


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



# re.compile('^SpamResults2_Dep_spam-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SpamResults2_Con_spam-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SpamResults2_Ran_spam-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SpamResults2_Std_spam-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SemResults2_Dep_sem-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SemResults2_Con_sem-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SemResults2_Ran_sem-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^SemResults2_Std_sem-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^IMDBResults2_Dep_imdb-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^IMDBResults2_Con_imdb-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^IMDBResults2_Ran_imdb-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^IMDBResults2_Std_imdb-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^HateResults2_Dep_hate-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^HateResults2_Con_hate-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^HateResults2_Ran_hate-rf\\-i\\-100_6_\\d+\\.pkl$')
# re.compile('^HateResults2_Std_hate-rf\\-i\\-100_6_\\d+\\.pkl$')


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
