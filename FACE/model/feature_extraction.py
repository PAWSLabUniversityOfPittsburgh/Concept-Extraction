from model.utils import *
from model.helpers import *

import math
import pickle
import spacy


class CandidateExtractor:
    """
    This class is to process data, select candidates based on patterns and extract features for each of the candidates
    """
    def init(self):
        pass

    def __init__(self, tag_pattern_path, config_path, verbose=True):
        self.doc = []
        self.doc_text = ""
        self.doc_id = 'null'
        self.phrase_count = dict()
        self.cur_idx = 0
        self.r = re.compile(get_regex_patterns(tag_pattern_path), flags=re.I)

        constant_variable, feature = load_config(config_path)
        self.longest_match_only = bool(constant_variable['LongestMatchOnly']['Value'])
        self.space_model = constant_variable['SpacyNLPModel']['Value']
        self.nlp = spacy.load(self.space_model)
        self.s = self.nlp("")
        self.N_MAX = constant_variable['MaxGram']['Value']
        self.feature_min = constant_variable['FeatureMin']['Value']
        self.min_string_len = constant_variable['MinStringLeng']['Value']

        self.feature_module_names = []
        self.feature_tags = []
        self.feature_types = []
        self.feature_paths = []
        self.corpus_statistic = None
        self.doc_leng = 0

        for key, value in feature.items():
            if value['Using'] == 'True':
                self.feature_module_names.append(value['ModuleName'])
                if value['ModuleName'] in ['feature_pos', 'feature_length_pos']:
                    for i in range(self.N_MAX):
                        self.feature_tags.append(value['Tag'] + '_' + str(i+1))
                        self.feature_types.append(value['Type'])
                        self.feature_paths.append(value['Path'])
                else:
                    self.feature_tags.append(value['Tag'])
                    self.feature_types.append(value['Type'])
                    self.feature_paths.append(value['Path'])

                    if self.corpus_statistic is None and value['ModuleName'] in ['feature_tfc', 'feature_tfidf', 'feature_okapi']:
                        with open(value['Path'], 'rb') as f:
                            self.corpus_statistic = pickle.load(f)

        self.df_column_names = ['gram', 'idx', 'tag', 'start_end', 'start_end_sen', 'doc_id'] + self.feature_tags
        self.final_cand_df = pd.DataFrame(columns=self.df_column_names)

        # ============================
        if verbose is True:
            print("[INFO] - Spacy model:", self.space_model)
            print("[INFO] - Maximum gram:", self.N_MAX)
            print("[INFO] - Minimum value of features:", self.feature_min)
            print("[INFO] - Minimum string length:", self.min_string_len)
            print("[INFO] - Features:", self.feature_tags)

    def set_document(self, doc):
        self.doc_id, self.doc_text, self.doc = doc[0], doc[1], sent_tokenize(doc[1])
        self.doc_leng = len(doc[1].split())
        self.phrase_count = dict()
        self.count_ngram_occurrences()

    def count_ngram_occurrences(self):
        for sen in self.doc:
            s = self.nlp(sen)
            for n in range(self.N_MAX):
                for i in range(len(s) - n):
                    tags = [s[j].tag_ for j in range(i, i + n + 1)]
                    if self.r.match(' '.join(tags)) is not None:
                        phrase = (str(s[i:(i + n)]) + ' ' + str(s[i+n].lemma_)).lstrip()
                        phrase = phrase.lower()
                        if phrase in self.phrase_count:
                            self.phrase_count[phrase] += 1
                        else:
                            self.phrase_count[phrase] = 1

    def extract_candidates(self):
        """
        Extract candidates for a whole document
        :return:
        """
        for sen in self.doc:
            self.final_cand_df = self.final_cand_df.append(self.extract_candidates_from_sentence(sen), ignore_index=True)
            # self.final_cand_df.append(self.extract_candidates_from_sentence(sen), ignore_index=True)

        print('Process doc: ', self.doc_id)

    def extract_candidates_from_sentence(self, sen):
        """
        Extract candidates from a sentence of the document
        :param sen: a sequence of words (string type)
        :return: <DataFrame> of candidates and their features
        """
        temp_cand_df = pd.DataFrame(columns=self.df_column_names)
        cur_idx = self.doc_text.find(sen)  # this variable to keep track of character index for the current sentence of a document
        self.s = self.nlp(sen)
        for n in range(self.N_MAX):
            for i in range(len(self.s) - n):
                if len(str(self.s[i:(i + n + 1)])) >= self.min_string_len and \
                        filter_containing_characters(str(self.s[i:(i + n + 1)])):

                    tags = [self.s[j].tag_ for j in range(i, i + n + 1)]
                    if self.r.match(' '.join(tags)) is not None:
                        feature_values = [self.s[i:(i + n + 1)], (i, i + n), [self.s[j].tag_ for j in range(i, i + n + 1)],
                                          (self.s[i].idx + cur_idx, self.s[i+n].idx + len(self.s[i+n]) - 1 + cur_idx),
                                          (self.s[i].idx, self.s[i+n].idx + len(self.s[i+n])), self.doc_id]

                        for fea_idx, feature in enumerate(self.feature_module_names):
                            if self.feature_paths[fea_idx] == "":
                                if feature in ['feature_pos', 'feature_length_pos']:
                                    feature_values.extend(getattr(self, feature)(feature_values))
                                else:
                                    feature_values.append(getattr(self, feature)(feature_values))
                            else:
                                feature_values.append(getattr(self, feature)(feature_values, self.feature_paths[fea_idx]))

                        temp_cand_df.loc[len(temp_cand_df)] = feature_values

        if self.longest_match_only is False:
            return temp_cand_df

        # filter candidates, only keep the longest ones
        final_cand_df = pd.DataFrame(columns=self.df_column_names)
        temp_len = len(temp_cand_df)
        for i in range(temp_len):
            flag = True
            temp_cand_idx = temp_cand_df['idx'][temp_len - 1 - i]
            for j in range(len(final_cand_df)):
                final_cand_idx = final_cand_df['idx'][j]
                if temp_cand_idx[0] >= final_cand_idx[0] and temp_cand_idx[1] <= final_cand_idx[1]:
                    flag = False
                    break

            if flag:
                final_cand_df.loc[len(final_cand_df)] = temp_cand_df.loc[temp_len - 1 - i]

        return final_cand_df

    def feature_length(self, feature_values):
        """

        :param feature_values: a list of [list_of_token_objects_in_spaCy_doc, (begin_word_position_in_sentence,
        end_word_position_in_sentence), list_of_token_tags, (begin_index_in_document, end_index_in_document),
        (begin_index_in_sentence, end_index_in_sentence), doc_id]
        :return: <int> - length of candidate
        """
        return len(feature_values[0])

    def feature_document_term_frequency(self, feature_values):
        """
        The term frequency of the candidate in the working document
        :param feature_values:
        :return: <int>
        """
        phrase = (str(feature_values[0][:len(feature_values[0])-1]) + ' ' +
                  str(feature_values[0][len(feature_values[0])-1].lemma_)).lstrip()
        return self.phrase_count[phrase.lower()]

    def feature_tfc(self, feature_values):
        """

        :param feature_values:
        :return: <int>
        """
        phrase = (str(feature_values[0][:-1]) + ' ' + str(feature_values[0][-1].lemma_)).lstrip()
        phrase = phrase.lower()
        return math.log(self.corpus_statistic['tfc'][phrase] if phrase in self.corpus_statistic['tfc']
                              else self.phrase_count[phrase])

    def feature_tfidf(self, feature_values):
        """

        :param feature_values:
        :return: <float>
        """
        phrase = (str(feature_values[0][:-1]) + ' ' + str(feature_values[0][-1].lemma_)).lstrip()
        phrase = phrase.lower()
        tf = self.phrase_count[phrase]
        idf = self.corpus_statistic['idf'][phrase] if phrase in self.corpus_statistic['idf'] else 1
        return (1 + math.log(tf))*math.log10(self.corpus_statistic['corpus_size']/idf)

    def feature_okapi(self, feature_values):
        """

        :param feature_values:
        :return: <float>
        """
        phrase = (str(feature_values[0][:-1]) + ' ' + str(feature_values[0][-1].lemma_)).lstrip()
        phrase = phrase.lower()
        tf = self.phrase_count[phrase]
        idf = self.corpus_statistic['idf'][phrase] if phrase in self.corpus_statistic['idf'] else 1

        k = 1.5
        b = 0.75
        tf_bm25 = (tf * (k + 1))/(tf + k * (1 - b + b * self.doc_leng/self.corpus_statistic['avgdl']))
        idf_bm25 = math.log((self.corpus_statistic['corpus_size'] - idf + 0.5)/(idf + 0.5))

        return tf_bm25*idf_bm25

    def feature_contain_named_entity(self, feature_values):
        """

        :param feature_values:
        :return: <'yes', 'no'>
        """
        for ent in self.s.ents:
            if ent.start_char >= feature_values[4][0] and ent.end_char <= feature_values[4][1]:
                return 'yes'

        return 'no'

    def feature_is_named_entity(self, feature_values):
        """

        :param feature_values:
        :return: <entity_label, 'no'>
        """
        for ent in self.s.ents:
            if ent.start_char == feature_values[4][0] and ent.end_char == feature_values[4][1]:
                return ent.label_

        return 'no'

    def feature_contain_stop_word(self, feature_values):
        """

        :param feature_values:
        :return: <int> - how many stop words
        """
        count = 0
        for token in feature_values[0]:
            if token.is_stop:
                count += 1

        return str(count)

    def feature_is_stop_word(self, feature_values):
        """

        :param feature_values:
        :return: <'yes', 'no'>
        """
        count = 0
        for token in feature_values[0]:
            if token.is_stop:
                count += 1

        return 'yes' if count == len(feature_values[0]) else 'no'

    def feature_start_end_with_stop_word(self, feature_values):
        """

        :param feature_values:
        :return: <'yes', 'no'>
        """

        return 'yes' if feature_values[0][0].is_stop or feature_values[0][-1].is_stop else 'no'

    def feature_contain_special_characters(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        special_chars = ['', '', '']

        for c in str(feature_values[0]):
            if c == '-':
                special_chars[0] = '-'
            elif c == '/':
                special_chars[1] = '/'
            elif c.isdigit():
                special_chars[2] = 'n'

        return ''.join(special_chars)

    def feature_pos_all(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        return '_'.join(feature_values[2])

    def feature_length_pos(self, feature_values):
        """

        :param feature_values:
        :return: <string> - concatenation of length and POS
        """
        pos = [''] * self.N_MAX
        leng = len(feature_values[2])
        for i in range(leng):
            pos[i] = str(leng) + '_' + feature_values[2][i]

        return pos

    def feature_pos(self, feature_values):
        """
        Get pos tag of each of the tokens of the candidate
        :param feature_values:
        :return: a list of pos tags
        """
        pos = ['']*self.N_MAX
        pos[:len(feature_values[2])] = feature_values[2][:]

        return pos

    def feature_word_1_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """

        return str(self.s[feature_values[1][0] - 1]) if feature_values[1][0] > 0 else '<start>'

    def feature_word_1_right(self, feature_values):
        """

        :param feature_values:
        :return:
        """

        return str(self.s[feature_values[1][1]+1]) if (feature_values[1][1] + 1) < len(self.s) else '<end>'

    def feature_word_2_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][0] == 0:
            return ''
        elif feature_values[1][0] == 1:
            return '<start>'
        else:
            return str(self.s[feature_values[1][0] - 2])

    def feature_word_2_right(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][1] + 1 == len(self.s):
            return ''
        elif feature_values[1][1] + 2 == len(self.s):
            return '<end>'
        else:
            return str(self.s[feature_values[1][1]+2])

    def feature_word_3_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][0] == 0:
            return ''
        elif feature_values[1][0] == 1:
            return ''
        elif feature_values[1][0] == 2:
            return '<start>'
        else:
            return str(self.s[feature_values[1][0] - 3])

    def feature_word_3_right(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][1] + 1 == len(self.s):
            return ''
        elif feature_values[1][1] + 2 == len(self.s):
            return ''
        elif feature_values[1][1] + 3 == len(self.s):
            return '<end>'
        else:
            return str(self.s[feature_values[1][1]+3])

    def feature_pos_1_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """

        return str(self.s[feature_values[1][0] - 1].tag_) if feature_values[1][0] > 0 else '<start>'

    def feature_pos_1_right(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """

        return str(self.s[feature_values[1][1]+1].tag_) if (feature_values[1][1] + 1) < len(self.s) else '<end>'

    def feature_pos_2_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][0] == 0:
            return ''
        elif feature_values[1][0] == 1:
            return '<start>'
        else:
            return str(self.s[feature_values[1][0] - 2].tag_)

    def feature_pos_2_right(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][1] + 1 == len(self.s):
            return ''
        elif feature_values[1][1] + 2 == len(self.s):
            return '<end>'
        else:
            return str(self.s[feature_values[1][1]+2].tag_)

    def feature_pos_3_left(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][0] == 0:
            return ''
        elif feature_values[1][0] == 1:
            return ''
        elif feature_values[1][0] == 2:
            return '<start>'
        else:
            return str(self.s[feature_values[1][0] - 3].tag_)

    def feature_pos_3_right(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        if feature_values[1][1] + 1 == len(self.s):
            return ''
        elif feature_values[1][1] + 2 == len(self.s):
            return ''
        elif feature_values[1][1] + 3 == len(self.s):
            return '<end>'
        else:
            return str(self.s[feature_values[1][1]+3].tag_)

    def feature_shape(self, feature_values):
        """

        :param feature_values:
        :return: <string>
        """
        return '_'.join([term.shape_ for term in feature_values[0]])

    def feature_c_value(self, feature_values):
        """
        Calculate c_value
        :param feature_value:
        :return: <float>
        """
        phrase = (str(feature_values[0][:len(feature_values[0]) - 1]) + ' ' +
                  str(feature_values[0][len(feature_values[0]) - 1].lemma_)).lstrip()
        fre = self.phrase_count[phrase.lower()]
        phrase_count_dic = get_phrase_count_dic_for_c_value(self.phrase_count, phrase)

        if len(phrase_count_dic) == 0:
            return math.log(len(feature_values[0]) + 1, 2)*fre
        else:
            second_term = 1/(len(phrase_count_dic))
            s = 0
            for v in phrase_count_dic.values():
                s += v

            second_term = second_term*s
            second_term = fre - second_term

            return math.log(len(feature_values[0]) + 1, 2)*second_term

    def feature_existing_phrase(self, feature_values, path):
        """
        Get a list of phrases from a file in the format of a list "[phrase1, phrase2, etc.]" per row
        :param feature_values:
        :param path:
        :return: the list
        """
        # Need to move loading files outside for loading only one time (i.e., init function)
        pass

    def feature_capitalized_all(self, feature_values):
        """

        :param feature_values:
        :return: <'yes', 'no'>
        """
        return 'yes' if str(feature_values[0]).isupper() else 'no'

    def feature_capitalized_one(self, feature_values):
        """

        :param feature_values:
        :return: <'yes', 'no'>
        """
        for token in feature_values[0]:
            if str(token).isupper():
                return 'yes'

        return 'no'
