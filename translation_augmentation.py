from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser("Script for extending train dataset")
parser.add_argument("--train_file_path", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv", required=False)
parser.add_argument("--languages", nargs="+", default=["es", "de", "fr", "zh"], required=False)
parser.add_argument("--thread-count", type=int, default=100, required=False)
parser.add_argument("--result-path", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/", required=False)

NAN_WORD = "_NAN_"


def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        time.sleep(1)
        text = text.translate(to="en")
        time.sleep(1)
    except NotTranslated:
        pass

    return str(text)


def test(args):
    
    train_data = pd.read_csv(args.train_file_path).iloc[:10]
    question_title = train_data["question_title"].fillna(NAN_WORD).values
    question_body = train_data["question_body"].fillna(NAN_WORD).values
    question_answer = train_data["answer"].fillna(NAN_WORD).values

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    parallel = Parallel(args.thread_count, backend="threading", verbose=5)
    for language in args.languages:
        print('Translate comments using "{0}" language'.format(language))
        
        translated_title = parallel(delayed(translate)(title, language) for title in question_title)
        train_data["title_" + language] = translated_title
        
        translated_body = parallel(delayed(translate)(body, language) for body in question_body)
        train_data["body_" + language] = translated_body
        
        translated_answer  = parallel(delayed(translate)(answer , language) for answer  in question_answer)
        train_data["answer _" + language] = translated_answer 

    result_path = os.path.join(args.result_path, "train_translate.csv")
    train_data.to_csv(result_path, index=False)

def main(args):

    train_data = pd.read_csv(args.train_file_path)
    question_title = train_data["question_title"].fillna(NAN_WORD).values
    question_body = train_data["question_body"].fillna(NAN_WORD).values
    question_answer = train_data["answer"].fillna(NAN_WORD).values

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    parallel = Parallel(args.thread_count, backend="threading", verbose=5)
    for language in args.languages:
        print('Translate comments using "{0}" language'.format(language))
        
        translated_title = parallel(delayed(translate)(title, language) for title in question_title)
        train_data["title_" + language] = translated_title
        
        translated_body = parallel(delayed(translate)(body, language) for body in question_body)
        train_data["body_" + language] = translated_body
        
        translated_answer  = parallel(delayed(translate)(answer , language) for answer  in question_answer)
        train_data["answer _" + language] = translated_answer 

    result_path = os.path.join(args.result_path, "train_translate.csv")
    train_data.to_csv(result_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    # test(args)
    main(args)
