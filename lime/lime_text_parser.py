from lime.lime_text import *
import stanza
from tqdm import tqdm
import random as rand
import pickle as pkl

class TextParserDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.num_exps = {}
        self.all_exps = {}
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.indexed_string.word(x[0]),
                '-'.join(
                    map(str,
                        self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            #print(exp)
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp if x[0] < (self.indexed_string.num_words() + self.indexed_string.tot_sents)]
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        exp = self.all_exps[label]
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text =  self.num_exps[label] * (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        #print(f"num_exps: {self.num_exps[label]}")
        text = re.sub(r'[<>&]', '|', text)
        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]) for x in exp]
        #print(f"exp: {exp}")
        #print(f"exp: {exp}")
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        #print(f"all_occurences(1): {all_occurrences}")
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        #print(f"all_occurences(2): {all_occurrences}\n\n")
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class LimeTextParserExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 parsing_type="dependency",
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False,
                 word_level=False,
                 random_trees=20,
                 max_sample_size=1/2,
                 mask_method=1):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

#////////////////////////////////////////////////////////////////////////////

        def init_parser(self):
            stanza.download('en')
            if parsing_type == "dependency":
                self.parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
            elif parsing_type == "constituency":
                self.parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
            elif parsing_type == "random":
                self.parser = stanza.Pipeline(lang='en', processors='tokenize')
            else:
                raise ValueError("PARSING METHOD NOT AN OPTION, HALTING")
               
        self.mask_method = mask_method
        self.parser_type = parsing_type
        init_parser(self)
        self.random_trees = random_trees
        self.max_sample_size = max_sample_size
        self.word_level=word_level

#////////////////////////////////////////////////////////////////////////////


        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         word_level=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_trees=None,
                         mask_method=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        #print(f"text_instance: {text_instance}") #////////////////////////
        if mask_method == None:
            mask_method = self.mask_method
        if word_level == None:
            word_level = self.word_level
        if random_trees == None:
            random_trees = self.random_trees

        def dependent_highlights(exp, indexed_string):
            #print(f"Original single exp: {exp}")
            allx = []
            for x in exp:
                allx.append(x[0])
            allx_static = allx.copy()
            for x in allx_static:
                deps = indexed_string.get_dependents([x])
                #print(f"X: {x}, Dependents: {deps}")
                for d in deps:
                    if d in allx and d is not x:
                        #exp[allx.index(d)][1] += exp[allx.index(x)][1]
                        exp[allx.index(d)] = (exp[allx.index(d)][0], exp[allx.index(d)][1] + exp[allx.index(x)][1])
                    elif d is not x:
                        exp.append((d, exp[allx.index(x)][1]))
                        allx.append(d)
            exp = [x for x in exp if indexed_string.id_is_word(x[0])]
            #print(f"depend single exp: {exp}")
            return exp
        

        indexed_string = (IndexedStringParsed(text_instance, parser=self.parser, 
                                        parse_type=self.parser_type, word_level=word_level,
                                        random_trees=random_trees))
        
        #print(f"indexed_string.tokens: {indexed_string.tokens}")
        #print(f"indexed_string.as_list: {indexed_string.as_list}")
        #print(f"indexed_string.positions: {indexed_string.positions}")
        #print(f"indexed_string.string_start: {indexed_string.string_start}")

        domain_mapper = TextParserDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric, mask_method=mask_method)
        #print(f"data: {data}")
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        #print(f"ret_exp.intercept: {ret_exp.intercept}")
        #print(f"ret_exp.local_exp: {ret_exp.local_exp}")
        #print(f"ret_exp.score: {ret_exp.score}")
        #print(f"ret_exp.local_pred: {ret_exp.local_pred}")

        #print(indexed_string.parse_tree)
        #print(f"original exp: {ret_exp.local_exp[1]}")
        #self.individual_groups = individual_dep_groups(ret_exp.local_exp)

        def individual_dep_groups(exp, string):
            #print(f"ORIGINAL full exp: {exp}")
            all_exps = {}
            for label in labels:
                label_exp = exp[label]
                label_exp_arr = []
                for i, lexp in enumerate(label_exp):
                    label_exp_arr.append((dependent_highlights([lexp], string), lexp[1]))
                label_exp_arr = sorted(label_exp_arr, key=lambda x: -abs(x[1]))
                all_exps[label] = [x[0] for x in label_exp_arr]
            #print(f"FINAL individual dependent explanations: {all_exps}")
            return all_exps
                #/store each dependent group separately to be displayed/#
        
        def combine_all_exps(all_exps, main_exp, string):
            #print(f"main_exp: {main_exp}")
            actual_all_exp = {}
            for label in labels:
                label_exp_arr = all_exps[label]
                actual_all_exp[label] = main_exp[label]
                for i, exp in enumerate(label_exp_arr, start=1):
                    rest = list([(x[0] + i*string.num_words(), x[1]) for x in exp])
                    #print(f"token offset: {i} * {string.num_words()}")
                    #print(f"rest[{i}]: {rest}")
                    for r in rest:
                        actual_all_exp[label].append(r)
                #print(f"FINAL COMBINED explanation: {actual_all_exp[label]}")
            return actual_all_exp

        rest_exps = individual_dep_groups(ret_exp.local_exp, indexed_string)
        main_exp = {}
        for label in labels:
            main_exp[label] = dependent_highlights(ret_exp.local_exp[label], indexed_string)
            domain_mapper.num_exps[label] = len(main_exp[label]) + 1
            #print(f"MAIN COMBINED explanation: {main_exp[label]}")
        #print(f"ret_exp.local_exp[1]: {ret_exp.local_exp[1]}")
        domain_mapper.all_exps = combine_all_exps(rest_exps, main_exp, indexed_string)
        ret_exp.local_exp = main_exp
        #print(f"dependent exp: {ret_exp.local_exp[1]}")

        # ////////////////////////////////////////////////////////////////////////////////
        # /////////////// DEBUGGED UP TO HERE, LOOK AT OUTPUT CODE NEXT //////////////////
        # ^^^^^^^^^^^^^^^ ON GOD ^^^^^^^^ THIS IS TRUE AGAIN ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        return ret_exp

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine',
                                mask_method=1):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_features()
        sample = self.random_state.randint(0, int(doc_size * self.max_sample_size), num_samples - 1)
        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]
        for i, size in tqdm(enumerate(sample, start=1), "Sampling"):
            inactive = self.random_state.choice(features_range, size,
                                                replace=False)
            #data[i, inactive] = 0


    #////// METHOD 1 /////////////////////////////////////////////
            if mask_method == 1:
                data[i] = indexed_string.mask_dependents(inactive)
                inverse_data.append(indexed_string.inverse_removing(data[i]))
    #/////////////////////////////////////////////////////////////

    #////// METHOD 2 /////////////////////////////////////////////
            elif mask_method == 2:
                data[i, inactive] = 0
                inverse_data.append(indexed_string.inverse_removing(indexed_string.mask_dependents(inactive)))
    #/////////////////////////////////////////////////////////////


            #if i == 1:
            #    print(f"\n\nparse tree: {indexed_string.parse_tree}")
            #    print(f"inactive: {inactive}")
            #    print(f"masked dependents: {data[i]}")
            #    print(f"inverse_data: {inverse_data[-1]}")
        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances

class SavedExplanation(object):
    def __init__(self, name, path, desc=None, exp=None, load=True):
        if exp != None:
            domain_mapper = exp.domain_mapper
            indexed_string = domain_mapper.indexed_string
            self.standard = False
            try:
                indexed_string.tokens
            except:
                self.standard = True
            indexed_string_data = {
                "raw_string": indexed_string.raw,
                "as_list": indexed_string.as_list,
                "positions": indexed_string.positions,
                "as_np": indexed_string.as_np,
                "string_start": indexed_string.string_start
            }
            if not self.standard:
                indexed_string_data["tokens"] = indexed_string.tokens,
                indexed_string_data["parse_tree"] = indexed_string.parse_tree
                indexed_string_data["tot_sents"] = indexed_string.tot_sents
                indexed_string_data["parse_type"] = indexed_string.parse_type
                indexed_string_data["inverse_ids"] = indexed_string.inverse_ids
                indexed_string_data["word_level"] = indexed_string.word_level
            domain_data = {
                "indexed_string": indexed_string_data
            }
            if not self.standard:
                domain_data["all_exps"] = domain_mapper.all_exps
                domain_data["num_exps"] = domain_mapper.num_exps
            exp_data = {
                "top_labels": exp.top_labels,
                "intercept": exp.intercept,
                "local_exp": exp.local_exp,
                "score": exp.score,
                "local_pred": exp.local_pred,
                "class_names": exp.class_names,
                "random_state": exp.random_state,
                "predict_proba": exp.predict_proba,
                "domain_data": domain_data
            }
            self.data = {
                "name": name,
                "path": path,
                "desc": desc,
                "exp_data": exp_data,
                "is_standard": self.standard
            }
            with open(path + name + ".pkl", mode="wb+") as file:
                pkl.dump(self.data, file=file)
        else:
            try:
                with open(path + name + ".pkl", 'rb') as file:
                    self.data = pkl.load(file)
            except:
                print("File does not exist")
                return None
            exp_data = self.data["exp_data"]
            domain_data = exp_data["domain_data"]
            indexed_string_data = domain_data["indexed_string"]

            if load:
                if self.data["is_standard"]:
                    indexed_string = IndexedString(indexed_string_data["raw_string"])
                    domain_mapper = TextDomainMapper(indexed_string)
                else:
                    indexed_string = IndexedStringParsed(no_init=True)
                    (indexed_string.raw, indexed_string.parse_tree,
                    indexed_string.inverse_ids, indexed_string.as_list,
                    indexed_string.positions, indexed_string.as_np, 
                    indexed_string.string_start, indexed_string.parse_type,
                    indexed_string.tot_sents, indexed_string.tokens) = (indexed_string_data["raw_string"],
                                                    indexed_string_data["parse_tree"],
                                                    indexed_string_data["inverse_ids"],
                                                    indexed_string_data["as_list"],
                                                    indexed_string_data["positions"],
                                                    indexed_string_data["as_np"],
                                                    indexed_string_data["string_start"],
                                                    indexed_string_data["parse_type"],
                                                    indexed_string_data["tot_sents"],
                                                    indexed_string_data["tokens"])
                    domain_mapper = TextParserDomainMapper(indexed_string)
                    (domain_mapper.all_exps, domain_mapper.num_exps) = (domain_data["all_exps"],
                                                                        domain_data["num_exps"])

                exp = explanation.Explanation(domain_mapper=domain_mapper,
                                            class_names=exp_data["class_names"],
                                            random_state=exp_data["random_state"])
                (exp.intercept, exp.local_exp, exp.score,
                exp.local_pred, exp.top_labels, exp.predict_proba) = (exp_data["intercept"],
                                                                    exp_data["local_exp"],
                                                                    exp_data["score"],
                                                                    exp_data["local_pred"],
                                                                    exp_data["top_labels"],
                                                                    exp_data["predict_proba"])
                self.exp = exp
        
    def get_exp(self):
        return self.exp

    def get_desc(self):
        return self.data["desc"]
    
    def get_name(self):
        return self.data["name"]
    
    def get_path(self):
        return self.data["path"]
    


class IndexedStringParsed(object):
    """String with various indexes."""

    def __init__(self, raw_string=None, parser=None, parse_type="dependency", 
                 word_level=False, random_trees=20, no_init=False):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        if not no_init:
            self.raw = raw_string + "\n\n\n"
            self.parser = parser
            self.parse_type = parse_type
            self.random_trees = random_trees
            self.word_level = word_level
            self.parse_tree, self.tokens = self.get_parsing(raw_string, parser)  
            self.inverse_ids = self.get_inverse_ids()  
            self.num_feats = self.num_features()
            self.as_list, self.positions = self._segment_with_tokens(self.raw, self.tokens)    
            self.as_np = np.array(self.as_list)
            self.string_start = np.hstack(
                ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))

        #self.positions = [i for i, word in enumerate(self.as_np)]

        '''
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match
        '''
        '''

        def non_word(string):
            return string not in self.as_list
            
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)
        '''
        

    #////////////////////////////////////////////////////////////////////////////

    def get_parsing(self, text_instance, parser):
        def total_sentences(pipeline_out):
            return len([sent for sent in pipeline_out.sentences])
        def organize_parse(pipeline_out, parse_type):
            def id_offset(word_id, sentence, doc):
                if word_id == 0:
                    #print(sentence.sent_id)
                    return int(sentence.sent_id)
                offset = self.tot_sents - 1
                for sent_id in range(int(sentence.sent_id)):
                    offset += len(doc.sentences[sent_id].words)
                return word_id + offset
            def get_branches(pipeline_out):
                return {id_offset(word.id, sent, pipeline_out): (id_offset(word.head, sent, pipeline_out), id_offset(word.id, sent, pipeline_out)) for sent in pipeline_out.sentences for word in sent.words}
            def get_ids_depend(pipeline_out):
                return {id_offset(word.id, sent, pipeline_out): word.text for sent in pipeline_out.sentences for word in sent.words}    
            def tree_depend(allbranches):
                ids = list(allbranches.keys())
                #print(f"len(ids): {len(ids)}")
                i = 0
                tree = []
                dependents, leftovers = depsFromBranches(allbranches, 0, ids)
                tree.append(dependents)
                while i < ids[-1]:
                    i += 1
                    dependents, leftovers = depsFromBranches(allbranches, i, leftovers)
                    tree.append(dependents)
                return tree
            def depsFromBranches(allbranches, wordid, ids):
                dependents = []
                if len(ids) == 0:
                    return dependents, ids
                leftovers = ids.copy()
                for n in ids:
                    branch = allbranches[n]
                    if branch[0] == wordid:
                        dependents.append(branch[1])
                        leftovers.remove(n)
                return dependents, leftovers
            def get_ids_constit(clean_cont_depens):
                idx = 0
                id_dict = {}
                for vertex in clean_cont_depens:
                    if len(vertex[1]) == 0:
                        id_dict[idx] = vertex[0]
                    idx += 1
                return id_dict
        
            def clean_const_depends(roots):
                def constituent_depends(constituencies, enum=0, dep_list=[]):
                    next_layer = []
                    for const in constituencies:
                        for child in const.children:
                            num_kids = len(child.children)
                            deps = list(range(enum + 1, enum + num_kids + 1))
                            dep_list.append((child.label, deps))
                            next_layer.append(child)
                            enum += num_kids
                    if len(next_layer) > 0:
                        constituent_depends(next_layer, enum, dep_list)
                    return dep_list
                def constituent_depends2(root, enum=1, dep_list=[]):
                    def how_many_kids(root):
                        children = 1
                        if len(root.children) > 0:
                            for child in root.children:
                                children += how_many_kids(child)
                        return children
                        
                    deps = []
                    enum_temp = enum
                    to_visit = root.children
                    for child in to_visit:
                        deps.append(enum_temp)
                        enum_temp += how_many_kids(child)
                    dep_list.append((root.label, deps))
                    for child in to_visit:
                        constituent_depends2(child, enum + 1, dep_list)
                        enum += how_many_kids(child)
                    return dep_list

                all_cont_dep = []
                for snum, sent in enumerate(roots.sentences):
                    #cont_dep = constituent_depends([sent.constituency], dep_list=[])
                    cont_dep = constituent_depends2(sent.constituency, dep_list=[])
                    y = [i for i in list(range(len(cont_dep))) if len(cont_dep[i][1]) == 1]
                    while len(y) > 0:
                        to_remove = []
                        for i in y:
                            if cont_dep[i][1][0] not in y:
                                to_remove.append(cont_dep[i][1][0])
                                cont_dep[i] = cont_dep[to_remove[-1]]#(cont_dep[to_remove[-1]][0], [])
                        to_remove = sorted(to_remove, reverse=True)
                        shifts = np.zeros(len(cont_dep), dtype=np.int16)
                        for r in to_remove:
                            shifts[r:] -= 1
                            cont_dep.pop(r)
                        for i in list(range(len(cont_dep))):
                            for j in list(range(len(cont_dep[i][1]))):
                                cont_dep[i][1][j] += shifts[cont_dep[i][1][j]]
                        y = [i for i in list(range(len(cont_dep))) if len(cont_dep[i][1]) == 1]
                    offset = 0
                    for s in range(len(all_cont_dep)):
                        offset += len(all_cont_dep[s])
                    if offset > 0:
                        for di, dep in enumerate(cont_dep):
                            newchildren = []
                            for child in dep[1]:
                                newchildren.append(child + offset)
                            cont_dep[di] = (dep[0], newchildren)
                    all_cont_dep.append(cont_dep.copy())
                appended_conts = []
                for di in all_cont_dep:
                    for dj in di:
                        appended_conts.append(dj)
                return appended_conts
            def just_tree(clean_cont_depens):
                return [x[1] for x in clean_cont_depens]
            def add_idv_words(branches, id_dict):
                new_word_dict = {}
                i = max(list(id_dict.keys()))
                for word_id in list(id_dict.keys()):
                    if len(branches[word_id]) > 0:
                        i += 1
                        new_word_dict[i] = id_dict[word_id]
                        branches[i] = (word_id, i)
                    else:
                        new_word_dict[word_id] = id_dict[word_id]
                return branches, new_word_dict
                        
            if self.parse_type == "dependency":
                branches = get_branches(pipeline_out)
                id_dict = get_ids_depend(pipeline_out)
                if self.word_level:
                    branches, id_dict = add_idv_words(branches, id_dict)
                parse_tree = tree_depend(branches)
            elif self.parse_type == "constituency":
                clean_deps = clean_const_depends(pipeline_out)
                id_dict = get_ids_constit(clean_deps)
                parse_tree = just_tree(clean_deps)
            return parse_tree, id_dict

        def custom_parse(text_instance):
            def random_connections(leaves, connections=[]):
                len_leaves = len(leaves)
                connections.append(leaves)

                if len_leaves == 1:
                    return connections

                n_groups = rand.randint(1, len_leaves - 1)

                assigns = []
                while not np.all([group in assigns for group in range(n_groups)]):
                    assigns = np.random.randint(0, n_groups, len_leaves)

                new_connections = [[] for _ in range(n_groups)]
                for i, group in enumerate(assigns):
                    new_connections[group].append(i)
                
                return random_connections(new_connections, connections)

            def random_tree(words):
                connections = list(reversed(random_connections(words, [])))
                
                branches = [[0 for _ in group] if type(group) is not str else [] for layer in connections for group in layer]
                word_dict = {}
                idx = 0
                len_prev = 0

                for i, layer in enumerate(connections):

                    len_prev += len(layer)

                    for j, group in enumerate(layer):
                        new_deps = []

                        if type(group) is not str:
                            new_deps = [g + len_prev for g in group]
                        else:
                            word_dict[idx] = group

                        branches[idx] = new_deps
                        idx += 1

                return branches, word_dict
            
            
            def combine_similar(parse_tree_tuples):
                
                def shift_connections(parse_tree, word_offset, nw_offset, word_start):
                    for i, group in enumerate(parse_tree):
                        for j, _ in enumerate(group):
                            if parse_tree[i][j] >= word_start:
                                parse_tree[i][j] += word_offset
                            parse_tree[i][j] += nw_offset
                            parse_tree[i][j] += 1
                    return parse_tree

                parse_trees = [tup[0] for tup in parse_tree_tuples]
                word_dicts = [tup[1] for tup in parse_tree_tuples]

                tree_trunks = []
                tree_lengths = []

                for i, tree in enumerate(parse_trees):
                    tree_trunks.append(tree[:list(word_dicts[i].keys())[0]])
                    tree_lengths.append(len(tree_trunks[-1]))

                word_offsets = [np.sum(tree_lengths[i:]) for i in range(1, len(tree_lengths))]
                word_offsets.append(0)

                nw_offset = 0
                root = [1]

                for i, tree in enumerate(tree_trunks):
                    word_start = list(word_dicts[i].keys())[0]
                    shift_connections(tree, word_offsets[i], nw_offset, word_start)

                    nw_offset += len(tree)
                    root.append(nw_offset + 1)

                root.pop()
                combined_trees = [root]
                for k in [j for i in range(len(tree_trunks)) for j in tree_trunks[i]]:
                    combined_trees.append(k)

                new_word_dict = {i + 1 + word_offsets[0]: word_dicts[0][i] for i in word_dicts[0]}
                for word in new_word_dict:
                    combined_trees.append([])

                return combined_trees, new_word_dict
            
            def combine_different(parse_tree_tuples):
                parse_trees = [tup[0] for tup in parse_tree_tuples]
                word_dicts = [tup[1] for tup in parse_tree_tuples]

                offsets = []
                for tree in parse_trees:
                    offsets.append(len(tree))
                
                offsets = [np.sum([offsets[:i]]) + 1 for i in range(len(offsets))]
                offsets = list(np.array(offsets, dtype=np.int16))

                combined = [offsets.copy()]

                for i, tree in enumerate(parse_trees):
                    for group in tree:
                        for j, _ in enumerate(group):
                            group[j] += offsets[i]
                        combined.append(group)

                new_word_dict = {}
                for j in range(len(word_dicts)):
                    for i in list(word_dicts[j].keys()):
                        new_word_dict[i + offsets[j]] = word_dicts[j][i]

                return combined, new_word_dict

            def combine_ran_trees(sentences, sample_size):
                combined = []

                for sent in tqdm(sentences, "Sentences"):

                    parse_trees = []

                    for s in tqdm(range(sample_size), "Random Parse Trees"):

                        parse_trees.append(random_tree(sent))

                    combined.append(combine_similar(parse_trees))

                return combine_different(combined)

            parse = self.parser(text_instance)
            words = [[token.text for token in sent.words] for sent in parse.sentences]
            self.tot_sents = len(words)
            return combine_ran_trees(words, self.random_trees)

        if self.parse_type == "random":
            return custom_parse(text_instance) # <--- ADD SAMPLE SIZE THROUGHOUT
        else:
            parse = parser(text_instance)
            self.tot_sents = total_sentences(parse)
            return organize_parse(parse, self.parse_type)

#////////////////////////////////////////////////////////////////////////////

    def get_inverse_ids(self):
        token_keys = self.tokens.keys()
        inverse_ids = {}
        for i, tk in enumerate(token_keys):
            inverse_ids[tk] = i
        return inverse_ids

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_features(self):
        if self.parse_type == "dependency":
            #return self.tot_sents + self.num_words()
            return len(self.parse_tree)
        elif self.parse_type == "constituency":
            return len(self.parse_tree)
        elif self.parse_type == "random":
            return len(self.parse_tree)

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        if self.parse_type == "dependency":
            #return len(self.tokens)
            return self.num_features()
        elif self.parse_type == "constituency":
            return self.num_features()
        elif self.parse_type == "random":
            return self.num_features()

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        #print(id_)
        if self.parse_type == "dependency":
            #id_ = ((id_ - self.tot_sents) % self.num_words()) + self.tot_sents
            id_ = id_ % self.num_words()
        elif self.parse_type == "constituency":
            id_ = id_ % self.num_words()#list(self.tokens.keys())[id_ % len(self.tokens)]#self.feature_to_id(id_ % self.num_words())
        elif self.parse_type == "random":
            id_ = id_ % self.num_words()
        return self.tokens[id_]
    
    
    def id_is_word(self, id_):
        #if self.parser_type == "dependency":
        #    return id_ >= self.tot_sents
        #elif self.parser_type == "constituency":
        #    return id_ in self.tokens.keys()
        return id_ in self.tokens.keys()

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        #if self.bow:
        #    return self.string_start[self.positions[id_]]
        #else:
        i = 0
        if self.parse_type == "dependency":
            # i = int((id_ - self.tot_sents) / self.num_words())
            # id_ = (id_ - self.tot_sents) % (self.num_words())
            i = int(id_ / self.num_words())
            id_ = self.feature_to_id(id_ % self.num_words())
        elif self.parse_type == "constituency":
            i = int(id_ / self.num_words())
            id_ = self.feature_to_id(id_ % self.num_words())
        elif self.parse_type == "random":
            i = int(id_ / self.num_words())
            id_ = self.feature_to_id(id_ % self.num_words())
        start = self.string_start[[self.positions[id_]]]
        #print(f"string start offset: {i} * {(self.string_start[-1] + len(self.as_list[-1]))}")
        start += i*(self.string_start[-1] + len(self.as_list[-1]))
        return start
        #if not self.bow:
        #    return self.inverse_vocab
        #else:
        #    print("PLEASE FINISH THE string_positions METHOD in the idxstringparsed class; around line 339")
        
    def feature_to_id(self, feature_id):
        return self.inverse_ids[feature_id]

    def mask_dependents(self, words_to_remove):
        def dep_vector(tree, heads):
            toremove = set()
            for h in heads:
                toremove.add(h)
                self.dep_vector_recursive(tree, tree[h], toremove)
            return toremove

        def binaryVec(toRemove, total):
            binaryvec = np.ones(total, dtype=np.int8)
            for zero in toRemove:
                binaryvec[zero] = 0
            return binaryvec
        
        to_remove_deps = dep_vector(self.parse_tree, words_to_remove)
        bvec = binaryVec(to_remove_deps, len(self.parse_tree))
        mask = bvec#[self.tot_sents:]
        return mask
    
    def dep_vector_recursive(self, tree, heads, toremove):
        for h in heads:
            if h not in toremove:
                toremove.add(h)
                self.dep_vector_recursive(tree, tree[h], toremove)
    
    def get_dependents(self, head):
        deps = set()
        self.dep_vector_recursive(self.parse_tree, head, deps)
        return deps


    def inverse_removing(self, mask):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])
        """
        return ''.join([f"{self.tokens[v]} " for v in np.flatnonzero(mask) if self.id_is_word(v)])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        positions = []
        text_ptr = 0
        for key in tokens.keys():
            token = tokens[key]
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            positions.append(len(list_form))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form, positions

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]