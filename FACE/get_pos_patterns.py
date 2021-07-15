from model.feature_extraction import *

import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a keyphrase extractor with feature engineering')
    parser.add_argument('--text_file', default='./data/sample/train.texts.txt', help='path to training text file')
    parser.add_argument('--label_file', default='./data/sample/train.labels.txt', help='path to training label file')
    parser.add_argument('--config_file', default='./model/files/config.json', help='path to config file')
    parser.add_argument('--pattern_file', default='./model/files/pos_patterns.txt',
                        help='path to part-of-speech pattern file')

    args = parser.parse_args()

    print('[INFO] - Setting')
    print('[INFO] -', args)

    labels = get_labels_2(args.label_file)
    docs = list(read_dataset_2(args.text_file))

    extractor = CandidateExtractor(tag_pattern_path=args.pattern_file, config_path=args.config_file, verbose=
                                   False)

    update_pos_pattern(args.pattern_file, docs, labels, extractor.nlp, extractor.N_MAX)

    r = re.compile(get_regex_patterns(args.pattern_file), flags=re.I)
    candidates = []
    for doc in docs:
        doc_candidates = []
        sents = sent_tokenize(doc[1])
        for sen in sents:
            s = extractor.nlp(sen)
            for n in range(6):
                for i in range(len(s) - n):
                    tags = [s[j].tag_ for j in range(i, i + n + 1)]
                    if r.match(' '.join(tags)) is not None:
                        doc_candidates.append(str(s[i:(i + n + 1)]).lower())

        candidates.append(doc_candidates)

    true_pos = 0
    false_neg = 0
    miss_list = []
    for i, cands in enumerate(candidates):
        keyphrases = labels[docs[i][0]]
        for k in keyphrases:
            if k in cands:
                true_pos += 1
            else:
                false_neg += 1
                miss_list.append(k)

    print('\nRecall:', true_pos / (true_pos + false_neg))
    # print(miss_list)
