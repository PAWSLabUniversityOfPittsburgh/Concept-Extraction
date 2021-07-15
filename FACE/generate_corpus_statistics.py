from model.feature_extraction import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a keyphrase extractor with feature engineering')
    parser.add_argument('--input_file', default='./data/sample/train.texts.txt', help='path to training input file')
    parser.add_argument('--output_file', default='corpus', help='corpus name to the output file')
    parser.add_argument('--pattern_file', default='./model/files/pos_patterns.txt',
                        help='path to part-of-speech pattern file')
    parser.add_argument('--config_file', default='./model/files/config.json', help='path to config file')

    args = parser.parse_args()

    print('[INFO] - Setting')
    print('[INFO] -', args)

    extractor = CandidateExtractor(tag_pattern_path=args.pattern_file, config_path=args.config_file, verbose=False)
    generate_tfc_and_idf_files(args.input_file, args.pattern_file, args.output_file, extractor.nlp, extractor.N_MAX)
