from model.feature_extraction import *

import time
import os
import argparse


start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a keyphrase extractor with feature engineering')
    parser.add_argument('--input_file', default='./data/sample/test.texts.txt', help='path to input file')
    parser.add_argument('--output_file', default='./output/output.txt', help='path to output file')
    parser.add_argument('--config_file', default='./model/files/config.json', help='path to config file')
    parser.add_argument('--pattern_file', default='./model/files/pos_patterns.txt', help='path to part-of-speech pattern file')
    parser.add_argument('--feature_file', default='./checkpoint/features.pkl', help='path to feature file')
    parser.add_argument('--model_file', default='./checkpoint/logreg.model', help='path to model file')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for some classifiers')

    args = parser.parse_args()

    print('[INFO] - Setting')
    print('[INFO] -', args)

    docs = list(read_dataset_2(args.input_file))
    # print(docs)
    extractor = CandidateExtractor(tag_pattern_path=args.pattern_file, config_path=args.config_file)

    for doc in docs:
        extractor.set_document(doc)
        extractor.extract_candidates()

    phrases, svm_rows = transform_to_svmlight_predicting(extractor, args.feature_file)

    with open('temp.svmlight', 'w') as fw:
        for row in svm_rows:
            fw.write(row)
            fw.write('\n')

    # Load a trained model
    model = pickle.load(open(args.model_file, 'rb'))
    X = get_data('temp.svmlight')[0]

    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    if args.threshold != 0.5:
        for i, prob in enumerate(y_pred_prob):
            y_pred[i] = 1 if prob >= args.threshold else 0

    # GROUP PHRASES BY THEIR IDS
    docid_phrases = dict()
    for i, v in enumerate(y_pred):
        if v == 1:
            if phrases[i][1] not in docid_phrases:
                docid_phrases[phrases[i][1]] = [(phrases[i][0], phrases[i][2])]
            else:
                docid_phrases[phrases[i][1]] = docid_phrases[phrases[i][1]] + [(phrases[i][0], phrases[i][2])]

    with open(args.output_file, 'w') as fw:
        for k, v in docid_phrases.items():
            fw.write(k + '\t' + str(v))
            fw.write('\n')

    os.remove("temp.svmlight")

print("\nRunning time:", time.time()-start_time)