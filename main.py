import jsonlines
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
from tqdm import tqdm


def get_score(vec_11, vec_22):
    dot_score = util.dot_score(vec_11, vec_22)[0].tolist()[0]
    cos_sim = util.cos_sim(vec_11, vec_22)[0].tolist()[0]
    euclidean = np.linalg.norm(vec_22 - vec_11)
    return dot_score, cos_sim, euclidean


def get_quora_q_file(
        filename="./datasets/quora_duplicate_questions.tsv",
        model=None,
        filename_answers="./results/quora_smallests.csv"
):
    with open(filename, 'r', encoding='utf-8') as tsvfile:
        # Iterate through each line in the TSV file

        total_lines = sum(1 for _ in tsvfile)
        tsvfile.seek(0)

        next(tsvfile)
        with open(filename_answers, 'w', newline='', encoding='utf-8') as csvfile:
            field_names = ["first_sentence", "second_sentence", "is_duplicate", "dot_score", "cos_score", "eucl_score"]

            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for line in tqdm(tsvfile, total=total_lines - 1):
                # Split the line by tabs to separate columns
                columns = line.strip().split('\t')
                # id_line = columns[0]
                first_sentence = columns[3]
                second_sentence = columns[4]
                is_duplicate = columns[5]
                first_emb = model.encode(first_sentence)
                second_emb = model.encode(second_sentence)

                dot_score, cos_score, eucl_score = get_score(first_emb, second_emb)
                answer_row = {"first_sentence": first_sentence, "second_sentence": second_sentence,
                              "is_duplicate": is_duplicate, "dot_score": dot_score, "cos_score": cos_score,
                              "eucl_score": eucl_score}
                writer.writerow(answer_row)


def get_sts_test(
        filename="./datasets/stsbenchmark/sts-test.csv",
        model=None,
        filename_answers="./results/sts_test_smallest.csv"
):
    with open(filename, 'r', encoding='utf-8') as csv_read:
        # Iterate through each line in the TSV file
        # csv_reader = csv.DictReader(csv_read)
        with open(filename_answers, 'w', newline='', encoding='utf-8') as csvfile:

            field_names = ["first_sentence", "second_sentence", "score", "dot_score", "cos_score", "eucl_score"]

            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for row in tqdm(csv_read):
                # Split the line by tabs to separate columns
                columns = row.strip().split('\t')

                first_sentence = columns[5]
                second_sentence = columns[6]
                score = columns[4]
                first_emb = model.encode(first_sentence)
                second_emb = model.encode(second_sentence)

                dot_score, cos_score, eucl_score = get_score(first_emb, second_emb)
                answer_row = {"first_sentence": first_sentence, "second_sentence": second_sentence,
                              "score": score, "dot_score": dot_score, "cos_score": cos_score,
                              "eucl_score": eucl_score}
                writer.writerow(answer_row)


def get_wnli_val(
        filename="./datasets/wnli/validation.jsonl",
        model=None,
        filename_answers="./results/wnli_val_smallest.csv"
):
    with jsonlines.open(filename) as reader:
        # Iterate through each line in the JSONL file
        total_lines = sum(1 for _ in reader)
    with jsonlines.open(filename) as reader:
        # Iterate through each line in the TSV file
        # csv_reader = csv.DictReader(csv_read)
        with open(filename_answers, 'w', newline='', encoding='utf-8') as csvfile:

            field_names = ["first_sentence", "second_sentence", "is_entailment", "dot_score", "cos_score", "eucl_score"]

            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for line in tqdm(reader, total=total_lines - 1):
                # Split the line by tabs to separate columns

                first_sentence = line['text1']
                second_sentence = line['text2']
                is_entailment = line['label']
                first_emb = model.encode(first_sentence)
                second_emb = model.encode(second_sentence)

                dot_score, cos_score, eucl_score = get_score(first_emb, second_emb)
                answer_row = {"first_sentence": first_sentence, "second_sentence": second_sentence,
                              "is_entailment": is_entailment, "dot_score": dot_score, "cos_score": cos_score,
                              "eucl_score": eucl_score}
                writer.writerow(answer_row)


def normalize_scores_binary(file_path='./results/wnli_val_smallest.csv', dataset=None):

    # Initialize variables to hold max and min values
    max_value = float('-inf')  # Initialize max_value as negative infinity
    min_value = float('inf')  # Initialize min_value as positive infinity

    # Open the CSV file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row if it exists
        column_index = header.index('cos_score')
        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[column_index])

            # Update max and min values if necessary
            if column_value > max_value:
                max_value = column_value
            if column_value < min_value:
                min_value = column_value
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row if it exists
        column_index = header.index('cos_score')
        # norm_scores = []
        # Iterate through each row in the CSV file
        correct_values = 0
        incorrect_values = 0
        total_values = 0
        if dataset == 'wnli':
            score_name = 'is_entailment'
        elif dataset == 'quora':
            score_name = 'is_duplicate'
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[column_index])
            normalized_score = (column_value - min_value)/ (max_value-min_value)
            # norm_scores.append(normalized_score)
            if normalized_score >= 0.5:
                binary_score = 1
            else:
                binary_score = 0

            binary_answer = int(row[header.index(score_name)])
            if binary_answer != 0 and binary_answer != 1:
                print(f"Weird is_entailment: {binary_answer}")
            if binary_answer == binary_score:
                correct_values += 1
            else:
                incorrect_values += 1
            total_values += 1
    print(f"Dataset: {dataset}\nGotten {round(correct_values/total_values * 100,2)} % correct values and "
          f"{round(incorrect_values/total_values * 100,2)} % incorrect values."
          f"Correct: {correct_values}, incorrect: {incorrect_values}, total: {total_values}\n")


def normalize_scores_scale(file_path='./results/sts_test_smallest.csv'):

    # Initialize variables to hold max and min values
    max_value = float('-inf')  # Initialize max_value as negative infinity
    min_value = float('inf')  # Initialize min_value as positive infinity

    # Open the CSV file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row if it exists
        column_index = header.index('cos_score')
        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[column_index])

            # Update max and min values if necessary
            if column_value > max_value:
                max_value = column_value
            if column_value < min_value:
                min_value = column_value
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row if it exists
        column_index = header.index('cos_score')
        # norm_scores = []
        # Iterate through each row in the CSV file
        total_values = 0
        exactly_the_same = 0
        under_50 = 0
        under_25 = 0
        under_10 = 0
        under_5 = 0
        under_1 = 0
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[column_index])
            normalized_score = (column_value - min_value) / (max_value-min_value)
            # norm_scores.append(normalized_score)
            norm_score_scale = round(normalized_score / 2 * 10, 1)

            scale_answer = float(row[header.index('score')])

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
            # else:
            #     incorrect_values += 1
            total_values += 1
        print(f"Dataset: sts\nGotten \n\tUnder 50 % rel.: {round(under_50 / total_values * 100, 2)} %"
              f"\n\tUnder 25 % rel.: {round(under_25 / total_values * 100, 2)} %"
              f"\n\tUnder 10 % rel.: {round(under_10 / total_values * 100, 2)} %"
              f"\n\tUnder 5 % rel.: {round(under_5 / total_values * 100, 2)} %"
              f"\n\tUnder 1 % rel.: {round(under_1 / total_values * 100, 2)} %"
              f"\n\tThe same % rel.: {round(exactly_the_same / total_values * 100, 2)} %")


def dataset_preprocessing(
        dataset_filenames=None,
        new_dataset_filename="",
        dataset_file_type=None
):
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



if __name__ == '__main__':
    # Dataset pre-processing
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
    exit()
    # normalize_scores_scale()
    # normalize_scores_binary(dataset='wnli')
    # normalize_scores_binary(file_path='./results/quora_smallests.csv', dataset='quora')
    # normalize_scores_scale(file_path='./results/sts_test_best_q.csv')
    # normalize_scores_binary(file_path="./results/wnli_val_best_q.csv", dataset='wnli')
    # normalize_scores_binary(file_path='./results/quora_best_q.csv', dataset='quora')
    # exit()
    model_smallest = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    get_wnli_val(model=model_smallest, filename_answers="./results/del1.csv")
    get_sts_test(model=model_smallest, filename_answers="./results/del2.csv")
    get_quora_q_file(model=model_smallest, filename_answers="./results/del3.csv")
    exit()
    model_best_quality = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    get_wnli_val(model=model_best_quality, filename_answers="./results/wnli_val_best_q.csv")
    get_sts_test(model=model_best_quality, filename_answers="./results/sts_test_best_q.csv")
    get_quora_q_file(model=model_best_quality, filename_answers="./results/quora_best_q.csv")
    # model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    # model = SentenceTransformer('consciousAI/cai-lunaris-text-embeddings')
