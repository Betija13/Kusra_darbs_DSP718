import jsonlines
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
from tqdm import tqdm
import os


def get_score(vec_11, vec_22):
    dot_score = util.dot_score(vec_11, vec_22)[0].tolist()[0]
    cos_sim = util.cos_sim(vec_11, vec_22)[0].tolist()[0]
    euclidean = np.linalg.norm(vec_22 - vec_11)
    return dot_score, cos_sim, euclidean


def get_metric_scores(
        filename="./datasets/wnli/validation.jsonl",
        model=None,
        filename_answers=None,
        model_name=None
):
    if not filename_answers:
        if not model_name:
            filename_answers = f"./metric_scores/{filename.split('/')[-1].split('.')[0]}.csv"
        else:
            if not os.path.exists(f"./metric_scores/{model_name}"):
                os.makedirs(f"./metric_scores/{model_name}")
            filename_answers = f"./metric_scores/{model_name}/{filename.split('/')[-1].split('.')[0]}.csv"
    with open(filename, 'r', encoding='utf-8') as file:
        total_values = sum(1 for _ in file)
    with open(filename, 'r', newline='', encoding='utf-8') as csv_read:
        reader = csv.DictReader(csv_read)
        header = reader.fieldnames
        with open(filename_answers, 'w', newline='', encoding='utf-8') as csvfile:
            field_names = ["first_sentence", "second_sentence", "score", "dot_score", "cos_score", "eucl_score", "score_type", "score_name"]
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for row in tqdm(reader, total=total_values):

                first_sentence = row['first_sentence']
                second_sentence = row['second_sentence']
                score = row['score']
                score_type = row['score_type']
                score_name = row['score_name']
                first_emb = model.encode(first_sentence)
                second_emb = model.encode(second_sentence)

                dot_score, cos_score, eucl_score = get_score(first_emb, second_emb)
                answer_row = {"first_sentence": first_sentence, "second_sentence": second_sentence,
                              "score": score, "dot_score": dot_score, "cos_score": cos_score,
                              "eucl_score": eucl_score, "score_type": score_type, "score_name": score_name}
                writer.writerow(answer_row)


def dataset_preprocessing(
        dataset_filenames=None,
        new_dataset_filename="",
        dataset_file_type=None
):
    if dataset_file_type is None:
        dataset_file_type = dataset_filenames[0].split(".")[-1]
    with open(new_dataset_filename, 'w', newline='', encoding='utf-8') as csvfile:

        field_names = ["id", "first_sentence", "second_sentence", "score", "score_name", "score_type"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        idx = 0
        for dataset in dataset_filenames:
            if dataset_file_type == "jsonl":
                with jsonlines.open(dataset) as reader:
                    total_lines = sum(1 for _ in reader)

                with jsonlines.open(dataset) as reader:
                    for line in tqdm(reader, total=total_lines - 1):

                        first_sentence = line['text1']
                        second_sentence = line['text2']
                        score = line['label']
                        score_name = 'is_entailment'
                        score_type = 'binary'

                        answer_row = {"id": idx, "first_sentence": first_sentence, "second_sentence": second_sentence,
                                      "score": score, "score_name": score_name, "score_type": score_type}
                        writer.writerow(answer_row)
                        idx += 1
                        if idx >= 1000:
                            break
                reader.close()
            elif dataset_file_type == "csv":
                with open(dataset, 'r', encoding='utf-8') as csv_read:
                    total_lines = sum(1 for _ in csv_read)
                    csv_read.seek(0)
                    for row in tqdm(csv_read, total=total_lines - 1):
                        columns = row.strip().split('\t')
                        first_sentence = columns[5]
                        second_sentence = columns[6]
                        score = columns[4]
                        score_name = 'score'
                        score_type = 'scale_0_5'
                        answer_row = {"id": idx, "first_sentence": first_sentence, "second_sentence": second_sentence,
                                      "score": score, "score_name": score_name, "score_type": score_type}
                        writer.writerow(answer_row)
                        idx += 1
                        if idx >= 1000:
                            break
                csv_read.close()
            elif dataset_file_type == "tsv":
                with open(dataset, 'r', encoding='utf-8') as tsvfile:
                    total_lines = sum(1 for _ in tsvfile)
                    tsvfile.seek(0)
                    next(tsvfile)
                    for line in tqdm(tsvfile, total=total_lines - 1):
                        # Split the line by tabs to separate columns
                        columns = line.strip().split('\t')
                        # id_line = columns[0]
                        first_sentence = columns[3]
                        second_sentence = columns[4]
                        score = columns[5]
                        score_name = 'is_duplicate'
                        score_type = 'binary'

                        answer_row = {"id": idx, "first_sentence": first_sentence, "second_sentence": second_sentence,
                                      "score": score, "score_name": score_name, "score_type": score_type}
                        writer.writerow(answer_row)
                        idx += 1
                        if idx >= 1000:
                            break
                tsvfile.close()
            if idx >= 1000:
                break


def get_results(file_path=None, metric_type='cos_score'):

    max_value = float('-inf')
    min_value = float('inf')

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        header = csv_reader.fieldnames
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[metric_type])
            score_type = row['score_type']
            if column_value > max_value:
                max_value = column_value
            if column_value < min_value:
                min_value = column_value
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        header = csv_reader.fieldnames
        total_values = 0
        exactly_the_same = 0
        if score_type != 'binary':
            under_50 = 0
            under_25 = 0
            under_10 = 0
            under_5 = 0
            under_1 = 0
        else:
            incorrect_values = 0
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[metric_type])
            normalized_score = (column_value - min_value) / (max_value-min_value)
            if score_type == 'binary':
                if normalized_score >= 0.5:
                    binary_score = 1
                else:
                    binary_score = 0

                binary_answer = int(row['score'])
                binary_name = row['score_name']
                if binary_answer != 0 and binary_answer != 1:
                    print(f"Weird {binary_name}: {binary_answer}")
                if binary_answer == binary_score:
                    exactly_the_same += 1
                else:
                    incorrect_values += 1

            else:
                name, score_min, score_max = score_type.split("_")
                if name != 'scale':
                    print(f"Name: {name}")
                norm_score_scale = round(normalized_score * float(score_max)+float(score_min), 1)
                scale_answer = float(row['score'])
                diff = abs(norm_score_scale - scale_answer)
                rel_diff = diff/5 * 100
                if rel_diff <= 50:
                    under_50 += 1
                # if rel_diff <= 20:
                #     correct_values += 1
                if rel_diff <= 25:
                    under_25 += 1
                if rel_diff <= 10:
                    under_10 += 1
                if rel_diff <= 5:
                    under_5 += 1
                if rel_diff <= 1:
                    under_1 += 1
                if rel_diff == 0:
                    exactly_the_same += 1
            total_values += 1
        dataset = file_path.split("/")[-1].split(".")[0]
        model = file_path.split("/")[-2]
        if score_type == 'binary':
            print(f"\nDataset: {dataset}\nModel:{model}\nMetric: {metric_type}"
                  f"\n\tCorrect {round(exactly_the_same / total_values * 100, 2)} % "
                  f"\n\t\tCorrect values: {exactly_the_same}"
                  f"\n\tIncorrect: {round(incorrect_values / total_values * 100, 2)} % "
                  f"\n\t\tIncorrect values: {incorrect_values}")
        else:
            print(f"\nDataset: {dataset}\nModel:{model}\nMetric: {metric_type}"
                  f"\n\tUnder 50 % rel.: {round(under_50 / total_values * 100, 2)} %"
                  f"\n\tUnder 25 % rel.: {round(under_25 / total_values * 100, 2)} %"
                  f"\n\tUnder 10 % rel.: {round(under_10 / total_values * 100, 2)} %"
                  f"\n\tUnder 5 % rel.: {round(under_5 / total_values * 100, 2)} %"
                  f"\n\tUnder 1 % rel.: {round(under_1 / total_values * 100, 2)} %"
                  f"\n\tThe same % rel.: {round(exactly_the_same / total_values * 100, 2)} %")


if __name__ == '__main__':
    task = input("Which task to do? \n\t1 - data pre-processing\n\t2 - get cos, eucl., and dot-product scores"
                 "\n\t3 - get results\n")
    while task != '-1':
        # Dataset pre-processing
        if task == "1":
            dataset_preprocessing(
                dataset_filenames=["./datasets/wnli/validation.jsonl"],
                new_dataset_filename="./valid_datasets/entailment_wnli.csv"
            )
            dataset_preprocessing(
                dataset_filenames=["./datasets/quora_duplicate_questions.tsv"],
                new_dataset_filename="./valid_datasets/quora_questions.csv"
            )
            dataset_preprocessing(
                dataset_filenames=["./datasets/stsbenchmark/sts-test.csv"],
                new_dataset_filename="./valid_datasets/sts.csv"
            )
        # cos, eucl., and dot-product scores
        elif task == '2':
            files = os.listdir('./valid_datasets')
            model_smallest = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
            print("model_smallest")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_smallest,
                    model_name='smallest'
                )
            model_best_quality = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            print("model_best_quality")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_best_quality,
                    model_name='best_q'
                )
        # get results
        elif task == '3':
            model_folders = os.listdir('./metric_scores')
            for model_fold in model_folders:
                files = os.listdir(f'./metric_scores/{model_fold}')
                for file in files:
                    get_results(file_path=f'./metric_scores/{model_fold}/{file}',
                                metric_type='cos_score')
                    get_results(file_path=f'./metric_scores/{model_fold}/{file}',
                                metric_type='eucl_score')
                    get_results(file_path=f'./metric_scores/{model_fold}/{file}',
                                metric_type='dot_score')
        task = input(
            "Which task to do? \n\t1 - data pre-processing\n\t2 - get cos, eucl., and dot-product scores"
            "\n\t3 - get results\n")
    if task == '-1':
        print("Thank you!\nGoodbye! :)")

