from model.feature_extraction import *

import time
import os
import argparse
from sklearn import linear_model


start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a keyphrase extractor with feature engineering')
    parser.add_argument('--text_file', default='./data/sample/train.texts.txt', help='path to training text file')
    parser.add_argument('--label_file', default='./data/sample/train.labels.txt', help='path to training label file')
    parser.add_argument('--config_file', default='./model/files/config.json', help='path to config file')
    parser.add_argument('--pattern_file', default='./model/files/pos_patterns.txt', help='path to part-of-speech pattern file')
    parser.add_argument('--feature_file', default='./checkpoint/features.pkl', help='path to feature file')
    parser.add_argument('--model_file', default='./checkpoint/logreg.model', help='path to model file')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for some classifiers')
    parser.add_argument('--split_data', type=float, default=0.0,
                        help='test size of split data for evaluation (default - no evaluation')
    parser.add_argument('--cross_validation', type=int, default=0,
                        help='number of folds for evaluation (default - no evaluation')

    args = parser.parse_args()

    print('[INFO] - Setting')
    print('[INFO] -', args)

    svm_filename = 'train_data_points.svmlight'

    # DATA PREPARATION AND FEATURE EXTRACTION
    print('\nPROCESS DATA FOR TRAINING')

    docs = list(read_dataset_2(args.text_file))
    extractor = CandidateExtractor(tag_pattern_path=args.pattern_file, config_path=args.config_file)

    for doc in docs:
        extractor.set_document(doc)
        extractor.extract_candidates()

    phrases, svm_rows, pos_indices, neg_indices, fea_idx_dic = transform_to_svmlight(extractor,
                                                                                     get_labels_2(args.label_file))
    with open(args.feature_file, 'wb') as f:
        pickle.dump(fea_idx_dic, f)

    with open(svm_filename, 'w') as fw:
        for row in svm_rows:
            fw.write(row)
            fw.write('\n')

    # MODEL TRAINING
    print('\nTRAIN EXTRACTOR')
    output_filename = 'results/genia.predicted_phrases.phrase'
    X_train, y_train = get_data(svm_filename)

    print('[INFO] - Number of train points:', X_train.shape)
    print('[INFO] - Number of positive points:', sum(y_train), round(sum(y_train) / X_train.shape[0], 2))

    model = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, class_weight='balanced')

    # save the model to disk
    model.fit(X_train, y_train)
    pickle.dump(model, open(args.model_file, 'wb'))

    # MODEL EVALUATION
    print('\nEVALUATING MODEL PERFORMANCE')
    if args.split_data != 0.0:
        print('Split data evaluation: test ratio', args.split_data)
        split_data(X_train, y_train, model, args.threshold, args.split_data)

    if args.cross_validation != 0:
        print('\nCross validation: number of folds', args.cross_validation)
        cross_validation(X_train, y_train, model, args.threshold, args.cross_validation)

    os.remove(svm_filename)

print("\nRunning time:", time.time()-start_time)
