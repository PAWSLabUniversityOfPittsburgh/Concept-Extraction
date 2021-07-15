
def check_phrase_substring(p1, p2):
    """
    Check phrase 1 is a substring of phrase 2
    :param p1:
    :param p2:
    :return: True or False
    """
    tokens_1 = p1.split(' ')
    tokens_2 = p2.split(' ')

    for i in range(len(tokens_2)-len(tokens_1)+1):
        if tokens_1[0] == tokens_2[i] and tokens_1 == tokens_2[i:(i+len(tokens_1))]:
            return True

    return False


def get_phrase_count_dic_for_c_value(phrases, phrase):
    """

    :param phrases: a dictionary of lemmatized phrases and their counts in the document
    :param phrase: target phrase
    :return: how many time the target phrase is a substring of other phrases in the document
    """
    phrase_count = dict()
    for p in phrases.keys():
        if p != phrase and check_phrase_substring(phrase, p):
            phrase_count[p] = phrases[p]

    return phrase_count


def transform_to_svmlight(extractor, labels):
    """

    :param extractor:
    :param labels:
    :return:
    """
    feature_set = []
    for i, tag in enumerate(extractor.feature_tags):
        if extractor.feature_types[i].split(':')[0] == 'numerical':
            min_value = 0
            max_value = int(extractor.feature_types[i].split(':')[1])
            for j in range(min_value, max_value+1):
                count = 0
                for irow in range(len(extractor.final_cand_df)):
                    if extractor.final_cand_df[tag][irow] <= j:
                        count += 1
                feature_set.append((tag + ':' + str(j), count))
        else:
            dic = dict()
            for irow in range(len(extractor.final_cand_df)):
                if extractor.final_cand_df[tag][irow] != '':
                    fea = tag + ':' + extractor.final_cand_df[tag][irow]
                    dic[fea] = 1 if fea not in dic else dic[fea]+1

            for k, v in dic.items():
                feature_set.append((k, v))

    # Remove features that appear less than min times (default 1)
    feature_set = [fea for fea in feature_set if fea[1] >= extractor.feature_min]

    # Create feature-index dictionary
    fea_idx_dic = dict()
    for i, fea in enumerate(feature_set):
        fea_idx_dic[fea[0]] = i + 1

    # CREATE SVM FILE
    rows = []
    phrases = []
    idx = 0
    pos_indices = []
    neg_indices = []
    for irow in range(len(extractor.final_cand_df)):
        phrase = ((str(extractor.final_cand_df[extractor.final_cand_df.columns[0]][irow])),
                  extractor.final_cand_df['doc_id'][irow], extractor.final_cand_df['start_end'][irow])
        if phrase[0].lower() in labels[extractor.final_cand_df['doc_id'][irow]]:
            row = '1' + '\t'
            pos_indices.append(idx)
        else:
            row = '0' + '\t'
            neg_indices.append(idx)
        idx += 1

        phrases.append(phrase)

        for i, tag in enumerate(extractor.feature_tags):
            if extractor.feature_types[i].split(':')[0] == 'numerical':
                min_value = 0
                max_value = int(extractor.feature_types[i].split(':')[1])
                for j in range(min_value, max_value + 1):
                    fea = tag + ':' + str(j)
                    if fea in fea_idx_dic and extractor.final_cand_df[tag][irow] <= j:
                        row += str(fea_idx_dic[fea]) + ':1' + '\t'
            else:
                fea = tag + ':' + extractor.final_cand_df[tag][irow]
                if fea in fea_idx_dic:
                    row += str(fea_idx_dic[fea]) + ':1' + '\t'

        rows.append(row)

    return phrases, rows, pos_indices, neg_indices, fea_idx_dic


def transform_to_svmlight_testing(extractor, labels, feature_filename):
    """

    :param extractor:
    :param labels:
    :param feature_filename:
    :return:
    """
    # Load feature set file
    import pickle

    with open(feature_filename, 'rb') as f:
        fea_idx_dic = pickle.load(f)

    # CREATE SVM FILE
    rows = []
    phrases = []
    idx = 0
    for irow in range(len(extractor.final_cand_df)):
        phrase = ((str(extractor.final_cand_df[extractor.final_cand_df.columns[0]][irow])),
                  extractor.final_cand_df['doc_id'][irow], extractor.final_cand_df['start_end'][irow])
        if phrase[0].lower() in labels[extractor.final_cand_df['doc_id'][irow]]:
            row = '1' + '\t'
        else:
            row = '0' + '\t'
        idx += 1

        phrases.append(phrase)

        for i, tag in enumerate(extractor.feature_tags):
            if extractor.feature_types[i].split(':')[0] == 'numerical':
                min_value = 0
                max_value = int(extractor.feature_types[i].split(':')[1])
                for j in range(min_value, max_value + 1):
                    fea = tag + ':' + str(j)
                    if fea in fea_idx_dic and extractor.final_cand_df[tag][irow] <= j:
                        row += str(fea_idx_dic[fea]) + ':1' + '\t'
            else:
                fea = tag + ':' + extractor.final_cand_df[tag][irow]
                if fea in fea_idx_dic:
                    row += str(fea_idx_dic[fea]) + ':1' + '\t'

        rows.append(row)

    return phrases, rows


def transform_to_svmlight_predicting(extractor, feature_filename):
    """

    :param extractor:
    :param feature_filename:
    :return:
    """
    # Load feature set file
    import pickle

    with open(feature_filename, 'rb') as f:
        fea_idx_dic = pickle.load(f)

    feature_size = len(fea_idx_dic)

    # print(fea_idx_dic)
    # CREATE SVM FILE
    rows = []
    phrases = []
    idx = 0
    for irow in range(len(extractor.final_cand_df)):
        phrase = ((str(extractor.final_cand_df[extractor.final_cand_df.columns[0]][irow])),
                  extractor.final_cand_df['doc_id'][irow], extractor.final_cand_df['start_end'][irow])

        row = '1' + '\t'
        idx += 1

        phrases.append(phrase)

        for i, tag in enumerate(extractor.feature_tags):
            if extractor.feature_types[i].split(':')[0] == 'numerical':
                min_value = 0
                max_value = int(extractor.feature_types[i].split(':')[1])
                for j in range(min_value, max_value + 1):
                    fea = tag + ':' + str(j)
                    if fea in fea_idx_dic and extractor.final_cand_df[tag][irow] <= j:
                        row += str(fea_idx_dic[fea]) + ':1' + '\t'
            else:
                fea = tag + ':' + extractor.final_cand_df[tag][irow]
                if fea in fea_idx_dic:
                    row += str(fea_idx_dic[fea]) + ':1' + '\t'

        # Check if the last feature is included because it will cause errors when transform to svmlight file
        if (str(feature_size)+':1') not in row:
            row += str(feature_size)+':0' + '\t'

        rows.append(row)

    return phrases, rows


def generate_tfc_and_idf_files(corpus_path, pos_pattern_path, output_file, nlp, n_max=6):
    """

    :param corpus_path:
    :param pos_pattern_path:
    :param output_file: path to pickle-formatted output file
    :param nlp: spaCy nlp model
    :param n_max: maximum of phrase length
    :return:
    """
    import pickle
    import re
    from nltk import sent_tokenize
    from model.utils import read_dataset_2, get_regex_patterns

    print('Running...')
    docs = list(read_dataset_2(corpus_path))

    r = re.compile(get_regex_patterns(pos_pattern_path), flags=re.I)

    tfc = dict()
    idf = dict()

    doc_lengs = []

    count = 0
    for doc in docs:

        count += 1
        print('Process doc', count)
        sents = sent_tokenize(doc[1])
        unique_phrases = set()

        for sen in sents:
            s = nlp(sen)
            for n in range(n_max):
                for i in range(len(s) - n):
                    tags = [s[j].tag_ for j in range(i, i + n + 1)]
                    if r.match(' '.join(tags)) is not None:
                        phrase = (str(s[i:(i + n)]) + ' ' + str(s[i + n].lemma_)).lstrip()
                        phrase = phrase.lower()
                        unique_phrases.add(phrase)
                        if phrase in tfc:
                            tfc[phrase] += 1
                        else:
                            tfc[phrase] = 1
        for phrase in unique_phrases:
            if phrase in idf:
                idf[phrase] += 1
            else:
                idf[phrase] = 1

        doc_lengs.append(len(doc[1].split()))

    print('\nNumber of documents in the corpus:', len(doc_lengs))
    print('Number of n-grams in the corpus:', len(tfc))
    corpus_statistic = {'corpus_size': len(doc_lengs), 'avgdl': sum(doc_lengs) / len(doc_lengs), 'tfc': tfc, 'idf': idf}

    with open(output_file, 'wb') as f:
        pickle.dump(corpus_statistic, f)


def downsampple_negative_data_points(data, labels, pos_indices, neg_indices, ratio=0.5):
    """

    :param data:
    :param labels:
    :param pos_indices:
    :param neg_indices:
    :param ratio:
    :return:
    """
    import random

    pos_len = len(pos_indices)
    neg_len = len(neg_indices)
    neg_num = int(pos_len/ratio - pos_len)

    random.seed(8)
    new_neg_indices = random.sample(neg_indices, neg_num)
    new_neg_indices.sort()
    final_indices = sorted(pos_indices+new_neg_indices)

    downsampled_data = [data[i] for i in final_indices]
    downsampled_labels = [labels[i] for i in final_indices]

    return downsampled_data, downsampled_labels


# FILTERING FUNCTIONS: help to remove noisy candidates (true negative), since there are only few samples, it's hard for
# model to learn from features
def filter_containing_characters(s):
    """
    If the string contains any character
    :param s:
    :return:
    """
    for c in s:
        if c.isalpha():
            return True

    return False


def filter_not_stopword(cand):

    count = 0
    for token in cand:
        if token.is_stop:
            count += 1

    return False if count == len(cand) else True

# ================================


def split_data(X, y, model, threshold=0.5, size=0.2):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn import metrics
    from model.utils import report2dict

    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=size, random_state=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    if threshold != 0.5:
        for i, prob in enumerate(y_pred_prob):
            y_pred[i] = 1 if prob >= threshold else 0

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob, pos_label=1)
    perf_df = report2dict(classification_report(y_test, y_pred))
    result = list(perf_df[perf_df['label'] == '1.0'].iloc[0][1:4]) + [metrics.auc(fpr, tpr)]
    print('precision:', result[0], ', recall:', result[1], ', f1-score:', result[2], ', AUC:', result[3])


def cross_validation(X, y, model, threshold=0.5, number_of_folds=5):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import classification_report
    from sklearn import metrics
    from model.utils import report2dict

    y_pred = cross_val_predict(model, X, y, cv=number_of_folds)
    y_pred_prob = cross_val_predict(model, X, y, cv=number_of_folds, method='predict_proba')[:, 1]

    if threshold != 0.5:
        for i, prob in enumerate(y_pred_prob):
            y_pred[i] = 1 if prob >= threshold else 0

    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob, pos_label=1)
    perf_df = report2dict(classification_report(y, y_pred))
    result = list(perf_df[perf_df['label'] == '1.0'].iloc[0][1:4]) + [metrics.auc(fpr, tpr)]
    print('precision:', result[0], ', recall:', result[1], ', f1-score:', result[2], ', AUC:', result[3])
