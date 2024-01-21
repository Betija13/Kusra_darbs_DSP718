import jsonlines
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
from tqdm import tqdm
import os
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel, \
    XLNetTokenizer, XLNetModel
import torch
from prettytable import PrettyTable
import tensorflow as tf
import tensorflow_hub as hub
import shutil
import matplotlib.pyplot as plt

COLOR_END = '\033[0m'
COLOR_BLACK = '\033[30m'
COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_BLUE = '\033[34m'
COLOR_MAGENTA = '\033[35m'
COLOR_CYAN = '\033[36m'
COLOR_WHITE = '\033[37m'


def get_score(vec_11, vec_22):
    dot_score = util.dot_score(vec_11, vec_22)[0].tolist()[0]
    cos_sim = util.cos_sim(vec_11, vec_22)[0].tolist()[0]
    euclidean = np.linalg.norm(vec_22 - vec_11)
    return dot_score, cos_sim, euclidean


def get_BERT_emb(first_sentence, second_sentence, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(first_sentence, second_sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings1 = outputs.last_hidden_state.mean(dim=1)
    embeddings2 = outputs.last_hidden_state.mean(dim=1)
    return embeddings1, embeddings2


def get_x_BERT_emb(first_sentence, second_sentence, model, model_name):
    if model_name == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name == 'DistilBERT':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == 'XLNet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    inputs = tokenizer(first_sentence, second_sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings1 = outputs.last_hidden_state[:, 0, :]
    embeddings2 = outputs.last_hidden_state[:, 0, :]
    return embeddings1.numpy(), embeddings2.numpy()


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
            field_names = ["first_sentence", "second_sentence", "score", "dot_score", "cos_score", "eucl_score",
                           "score_type", "score_name"]
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for row in tqdm(reader, total=total_values):

                first_sentence = row['first_sentence']
                second_sentence = row['second_sentence']
                score = row['score']
                score_type = row['score_type']
                score_name = row['score_name']
                if model_name == "BERT":
                    first_emb, second_emb = get_BERT_emb(first_sentence, second_sentence, model)
                elif model_name == 'RoBERTa' or model_name == 'DistilBERT' or model_name == 'XLNet':
                    first_emb, second_emb = get_x_BERT_emb(first_sentence, second_sentence, model, model_name)
                elif model_name == 'USE':
                    first_emb, second_emb = model([first_sentence, second_sentence]).numpy()
                else:
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
    dataset_size = 500
    dataset_size += 1
    if dataset_file_type is None:
        dataset_file_type = dataset_filenames[0].split(".")[-1]
    with open(new_dataset_filename, 'w', newline='', encoding='utf-8') as csvfile:

        field_names = ["id", "first_sentence", "second_sentence", "score", "score_name", "score_type"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        idx = 0
        for dataset in dataset_filenames:
            folder_name = dataset.split("/")[-2]
            print(folder_name)
            if dataset_file_type == "jsonl":
                with jsonlines.open(dataset) as reader:
                    for line in tqdm(reader, total=dataset_size - 1):

                        first_sentence = line['text1']
                        second_sentence = line['text2']
                        score = line['label']
                        score_name = 'is_entailment'
                        score_type = 'binary'

                        answer_row = {"id": idx, "first_sentence": first_sentence, "second_sentence": second_sentence,
                                      "score": score, "score_name": score_name, "score_type": score_type}
                        writer.writerow(answer_row)
                        idx += 1
                        if idx >= dataset_size:
                            break
                reader.close()
            elif dataset_file_type == "csv":
                with open(dataset, 'r', encoding='utf-8') as csv_read:
                    next(csv_read)
                    for row in tqdm(csv_read, total=dataset_size - 1):
                        try:
                            if folder_name == 'bible_verses':

                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[0]
                                second_sentence = columns[1]
                                score = columns[2]
                                score_name = 'is_duplicate'
                                score_type = 'binary'

                            elif folder_name == 'home_depot':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[2]
                                second_sentence = columns[3]
                                score = columns[4]
                                score_name = 'score'
                                score_type = 'scale_0_5'
                            elif folder_name == 'messages':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[1]
                                second_sentence = columns[2]
                                score = columns[7]
                                score_name = 'score'
                                score_type = 'scale_0_1'
                            elif folder_name == 'qnli':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[0]
                                second_sentence = columns[1]
                                score = columns[2]
                                score_name = 'is_entailment'
                                score_type = 'binary'
                            elif folder_name == 'rte':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[0]
                                second_sentence = columns[1]
                                score = columns[2]
                                score_name = 'is_entailment'
                                score_type = 'binary'
                            elif folder_name == 'sem_sim_phones':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[1]
                                second_sentence = columns[2]
                                score = columns[3]
                                score_name = 'is_reason'
                                score_type = 'binary'
                            elif folder_name == 'stsbenchmark':
                                columns = row.strip().split('\t')
                                first_sentence = columns[5]
                                second_sentence = columns[6]
                                score = columns[4]
                                score_name = 'score'
                                score_type = 'scale_0_5'
                            elif folder_name == 'wnli':
                                columns = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
                                first_sentence = columns[0]
                                second_sentence = columns[1]
                                score = columns[2]
                                score_name = 'is_entailment'
                                score_type = 'binary'
                            else:
                                print(folder_name, " is in else")
                                columns = row.strip().split('\t')
                                first_sentence = columns[1]
                                second_sentence = columns[2]
                                score = columns[7]
                                score_name = 'score'
                                score_type = 'scale_0_5'
                            answer_row = {"id": idx, "first_sentence": first_sentence,
                                          "second_sentence": second_sentence,
                                          "score": score, "score_name": score_name, "score_type": score_type}
                            writer.writerow(answer_row)
                            idx += 1
                        except:
                            idx -= 1
                        if idx >= dataset_size:
                            break

                csv_read.close()
            elif dataset_file_type == "tsv":
                with open(dataset, 'r', encoding='utf-8') as tsvfile:
                    next(tsvfile)
                    for line in tqdm(tsvfile, total=dataset_size - 1):
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
                        if idx >= dataset_size:
                            break
                tsvfile.close()
            if idx >= dataset_size:
                break


def get_results(file_path=None, metric_type='cos_score'):
    score_dict = {}
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
            above_50 = 0
            above_67 = 0
            above_75 = 0
            above_90 = 0
            total_50 = 1e-8
            total_67 = 1e-8
            total_75 = 1e-8
            total_90 = 1e-8
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            column_value = float(row[metric_type])
            normalized_score = (column_value - min_value) / (max_value - min_value + 1e-8)
            if score_type == 'binary':
                binary_score_50 = None
                binary_score_67 = None
                binary_score_75 = None
                binary_score_90 = None
                if normalized_score >= 0.5:
                    binary_score_50 = 1
                elif normalized_score <= (1 - 0.5):
                    binary_score_50 = 0
                if normalized_score >= (2 / 3):
                    binary_score_67 = 1
                elif normalized_score <= (1 - (2 / 3)):
                    binary_score_67 = 0
                if normalized_score >= 0.75:
                    binary_score_75 = 1
                elif normalized_score <= (1 - 0.75):
                    binary_score_75 = 0
                if normalized_score >= 0.9:
                    binary_score_90 = 1
                elif normalized_score <= (1 - 0.9):
                    binary_score_90 = 0

                binary_answer = int(row['score'])
                binary_name = row['score_name']
                if binary_answer != 0 and binary_answer != 1:
                    print(f"Weird {binary_name}: {binary_answer}")
                if binary_score_50 is not None:
                    total_50 += 1
                    if binary_answer == binary_score_50:
                        above_50 += 1
                if binary_score_67 is not None:
                    total_67 += 1
                    if binary_answer == binary_score_67:
                        above_67 += 1
                if binary_score_75 is not None:
                    total_75 += 1
                    if binary_answer == binary_score_75:
                        above_75 += 1
                if binary_score_90 is not None:
                    total_90 += 1
                    if binary_answer == binary_score_90:
                        above_90 += 1

            else:
                name, score_min, score_max = score_type.split("_")
                if name != 'scale':
                    print(f"Name: {name}")
                norm_score_scale = round(normalized_score * float(score_max) + float(score_min), 1)
                scale_answer = float(row['score'])
                diff = abs(norm_score_scale - scale_answer)
                rel_diff = diff / (float(score_max) - float(score_min)) * 100
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
        if score_type == 'binary':
            print(f"\n\t\t{COLOR_CYAN}Metric: {metric_type}{COLOR_END}"
                  f"\n\t\t\tCorrect above 0.5  {round(above_50 / total_50 * 100, 2)} %, total: {round(above_50 / total_values * 100, 2)} % "
                  f"\n\t\t\tCorrect above 0.67 {round(above_67 / total_67 * 100, 2)} %, total: {round(above_67 / total_values * 100, 2)} % "
                  f"\n\t\t\tCorrect above 0.75 {round(above_75 / total_75 * 100, 2)} %, total: {round(above_75 / total_values * 100, 2)} % "
                  f"\n\t\t\tCorrect above 0.90 {round(above_90 / total_90 * 100, 2)} %, total: {round(above_90 / total_values * 100, 2)} % ")
            return_score = (round(above_50 / total_50 * 100, 2), round(above_67 / total_67 * 100, 2),
                            round(above_75 / total_75 * 100, 2), round(above_90 / total_90 * 100, 2),
                            round(above_50 / total_values * 100, 2), round(above_67 / total_values * 100, 2),
                            round(above_75 / total_values * 100, 2), round(above_90 / total_values * 100, 2))
        else:
            print(f"\n\t\t{COLOR_CYAN}Metric: {metric_type}{COLOR_END}"
                  f"\n\t\t\tUnder 50 % rel.: {round(under_50 / total_values * 100, 2)} %"
                  f"\n\t\t\tUnder 25 % rel.: {round(under_25 / total_values * 100, 2)} %"
                  f"\n\t\t\tUnder 10 % rel.: {round(under_10 / total_values * 100, 2)} %"
                  f"\n\t\t\tUnder 5 % rel.: {round(under_5 / total_values * 100, 2)} %"
                  f"\n\t\t\tUnder 1 % rel.: {round(under_1 / total_values * 100, 2)} %"
                  f"\n\t\t\tThe same % rel.: {round(exactly_the_same / total_values * 100, 2)} %")
            return_score = (round(under_50 / total_values * 100, 2), round(under_25 / total_values * 100, 2),
                            round(under_10 / total_values * 100, 2), round(under_5 / total_values * 100, 2),
                            round(under_1 / total_values * 100, 2), round(exactly_the_same / total_values * 100, 2))
    return return_score


def print_scores_as_table(score_dict):
    # Create a PrettyTable instance
    table = PrettyTable()

    # Define columns (files)
    columns = ['Model'] + list(next(iter(score_dict.values())).keys())
    table.field_names = columns

    # Add rows to the table
    for model, scores in score_dict.items():
        row = [model] + list(scores.values())
        table.add_row(row)

    # Print the table
    print(table)

def get_dataset_info(filename, save_pic=True):
    file = filename.split("/")[-1]
    with open(filename, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        header = csv_reader.fieldnames
        total_scores = []
        for row in csv_reader:
            # Convert the column value to float for numeric comparison
            if save_pic:
                score_value = float(row['score'])
                score_type = row['score_type']
            else:
                score_value = float(row['relevance'])
                score_type = 'scale_0_5'
            total_scores.append(score_value)
        if score_type == 'binary':
            count_of_zeros = total_scores.count(0)
            count_of_ones = total_scores.count(1)
            print(f"{COLOR_RED}{file} info: {COLOR_END}\n\t1: {count_of_ones}\t\tpercent: {round(count_of_ones/len(total_scores) * 100, 2)} %\n\t0: {count_of_zeros}\t\tpercent: {round(count_of_zeros/len(total_scores) * 100, 2)} %")
            plt.bar([0, 1], [count_of_zeros, count_of_ones], color=['blue', 'green'])
            plt.xticks([0, 1], ['0', '1'])
            plt.xlabel('Values')
            plt.ylabel('Count')
            plt.title(f'Distribution of {file.split(".")[0]} dataset')
            if save_pic:
                folder_path = 'dataset_info_histograms'
                os.makedirs(folder_path, exist_ok=True)
                image_path = os.path.join(folder_path, f'{file.split(".")[0]}_histogram.png')
                plt.savefig(image_path)
            plt.show()
        elif score_type.split("_")[0] == 'scale':
            max_value = max(total_scores)
            min_value = min(total_scores)

            num_bins = 5
            bin_width = (max_value - min_value) / num_bins
            bins = [min_value + i * bin_width for i in range(num_bins + 1)]

            hist_counts, bin_edges, _ = plt.hist(total_scores, bins=bins, color='skyblue', edgecolor='black')
            print(f"{COLOR_RED}{file} info: {COLOR_END}")
            for i, count in enumerate(hist_counts):
                range_start, range_end = bin_edges[i], bin_edges[i + 1]
                print(
                    f"Range {i + 1}: {range_start:.2f} to {range_end:.2f} - Count: {int(count)}\t\t percentage: {round(count / len(total_scores) * 100, 2)} %")

            plt.xlabel('Data Ranges')
            plt.ylabel('Count')
            plt.title(f'Distribution of {file.split(".")[0]} dataset')

            if save_pic:
                folder_path = 'dataset_info_histograms'
                os.makedirs(folder_path, exist_ok=True)
                image_path = os.path.join(folder_path, f'{file.split(".")[0]}_histogram.png')
                plt.savefig(image_path)
            plt.show()



if __name__ == '__main__':
    choices_string = "Which task to do? \n\t1 - data pre-processing\n\t2 - get cos, eucl., and dot-product scores" \
                     "\n\t3 - get results\n\t4 - model sizes\n\t5 - get info about datasets (histogramms)\n"
    task = input(choices_string)
    while task != '-1':
        # Dataset pre-processing
        if task == "1":
            # print(f"{COLOR_RED}You probably didn't mean to come here!{COLOR_END}")
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/bible_verses/testing_set.csv"],
            #     new_dataset_filename="./valid_datasets/bible_verses.csv"
            # )
            dataset_preprocessing(
                dataset_filenames=["./datasets/home_depot/train.csv"],
                new_dataset_filename="./valid_datasets/HD.csv"
            )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/messages/messageWithFeature.csv"],
            #     new_dataset_filename="./valid_datasets/messages.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/qnli/qnli_validation.csv"],
            #     new_dataset_filename="./valid_datasets/QNLI.csv"
            # )
            dataset_preprocessing(
                dataset_filenames=["./datasets/quora_question_pairs/quora_duplicate_questions.tsv"],
                new_dataset_filename="./valid_datasets/Quora.csv"
            )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/rte/rte_train.csv"],
            #     new_dataset_filename="./valid_datasets/rte.csv"
            # )
            dataset_preprocessing(
                dataset_filenames=["./datasets/sem_sim_phones/train - train.csv"],
                new_dataset_filename="./valid_datasets/SS.csv"
            )
            dataset_preprocessing(
                dataset_filenames=["./datasets/stsbenchmark/sts-train.csv"],
                new_dataset_filename="./valid_datasets/STSB.csv"
            )
            dataset_preprocessing(
                dataset_filenames=["./datasets/wnli/wnli_train.csv"],
                new_dataset_filename="./valid_datasets/WNLI.csv"
            )
        # cos, eucl., and dot-product scores
        elif task == '2':
            files = os.listdir('./valid_datasets')
            print(f"{COLOR_RED}You probably didn't mean to come here!{COLOR_END}")
            model_smallest = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
            print(f"{COLOR_GREEN}model_smallest{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_smallest,
                    model_name='ST_S'
                )
            model_best_quality = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            print(f"{COLOR_GREEN}model_best_quality{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_best_quality,
                    model_name='ST_B'
                )
            model_BERT = BertModel.from_pretrained('bert-base-uncased')
            print(f"{COLOR_GREEN}BERT{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_BERT,
                    model_name='BERT'
                )
            model_GloVe = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
            print(f"{COLOR_GREEN}GloVe{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_GloVe,
                    model_name='ST_F'
                )
            # model_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
            model_t5 = SentenceTransformer('sentence-transformers/sentence-t5-large')
            print(f"{COLOR_GREEN}t5{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_t5,
                    model_name='T5'
                )
            # model_RoBERTa = RobertaModel.from_pretrained('roberta-base')
            # print(f"{COLOR_GREEN}RoBERTa{COLOR_END}")
            # for file in files:
            #     print(file)
            #     get_metric_scores(
            #         filename=f"./valid_datasets/{file}",
            #         model=model_RoBERTa,
            #         model_name='RoBERTa'
            #     )
            # model_DistilBERT = DistilBertModel.from_pretrained('distilbert-base-uncased')
            # print(f"{COLOR_GREEN}DistilBERT{COLOR_END}")
            # for file in files:
            #     print(file)
            #     get_metric_scores(
            #         filename=f"./valid_datasets/{file}",
            #         model=model_DistilBERT,
            #         model_name='DistilBERT'
            #     )
            # model_XLNet = XLNetModel.from_pretrained('xlnet-base-cased')
            # print(f"{COLOR_GREEN}XLNet{COLOR_END}")
            # for file in files:
            #     print(file)
            #     get_metric_scores(
            #         filename=f"./valid_datasets/{file}",
            #         model=model_XLNet,
            #         model_name='XLNet'
            #     )
            model_USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            print(f"{COLOR_GREEN}USE{COLOR_END}")
            for file in files:
                print(file)
                get_metric_scores(
                    filename=f"./valid_datasets/{file}",
                    model=model_USE,
                    model_name='USE'
                )
            # model_SBERT = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            # print(f"{COLOR_GREEN}SBERT{COLOR_END}")
            # for file in files:
            #     print(file)
            #     get_metric_scores(
            #         filename=f"./valid_datasets/{file}",
            #         model=model_SBERT,
            #         model_name='SBERT'
            #     )
            # model_LaBSE = SentenceTransformer('sentence-transformers/LaBSE')
            # print(f"{COLOR_GREEN}LaBSE{COLOR_END}")
            # for file in files:
            #     print(file)
            #     get_metric_scores(
            #         filename=f"./valid_datasets/{file}",
            #         model=model_LaBSE,
            #         model_name='LaBSE'
            #     )
        # get results
        elif task == '3':
            models_to_use = ['BERT', 'ST_B', 'ST_F', 'ST_S', 'T5', 'USE']
            files_to_use = ['HD.csv', 'wnli.csv', 'SS.csv', 'STSB.csv', 'Quora.csv']
            scores_to_use = ['cos'] #['cos', 'eucl', 'dot']
            model_folders = os.listdir('./results')
            scores = {}
            for score in scores_to_use:
                scores[score] = {}
            for score_name in scores:
                scores[score_name]['binary'] = {"above_50": {}, "above_67": {}, "above_75": {}, "above_90": {},
                                                "total_above_50": {}, "total_above_67": {}, "total_above_75": {},
                                                "total_above_90": {}}
                scores[score_name]['scale'] = {"rel_50": {}, "rel_25": {}, "rel_10": {}, "rel_5": {}, "rel_1": {},
                                               "exact": {}}
            for model_fold in model_folders:
                if model_fold in models_to_use:
                    for score_name in scores_to_use:
                        scores[score_name]['scale']['rel_50'][model_fold] = {}
                        scores[score_name]['scale']['rel_25'][model_fold] = {}
                        scores[score_name]['scale']['rel_10'][model_fold] = {}
                        scores[score_name]['scale']['rel_5'][model_fold] = {}
                        scores[score_name]['scale']['rel_1'][model_fold] = {}
                        scores[score_name]['scale']['exact'][model_fold] = {}
                        scores[score_name]['binary']['above_50'][model_fold] = {}
                        scores[score_name]['binary']['above_67'][model_fold] = {}
                        scores[score_name]['binary']['above_75'][model_fold] = {}
                        scores[score_name]['binary']['above_90'][model_fold] = {}
                        scores[score_name]['binary']['total_above_50'][model_fold] = {}
                        scores[score_name]['binary']['total_above_67'][model_fold] = {}
                        scores[score_name]['binary']['total_above_75'][model_fold] = {}
                        scores[score_name]['binary']['total_above_90'][model_fold] = {}
                    print(f"\n{COLOR_MAGENTA}Model: {model_fold}{COLOR_END}")
                    files = os.listdir(f'./results/{model_fold}')
                    for file in files:
                        if file in files_to_use:
                            file_name_dict = file.split('.')[0]
                            print(f"\n\t{COLOR_BLUE}Dataset: {file_name_dict}{COLOR_END}")
                            if file == 'HD.csv' or file == 'STSB.csv':
                            # if file == 'home_depot.csv' or file == 'messages.csv' or file == 'stsbenchmark.csv':
                                for score_name in scores_to_use:
                                    score_50, score_25, score_10, score_5, score_1, score_exact = get_results(
                                        file_path=f'./results/{model_fold}/{file}',
                                        metric_type=f'{score_name}_score')
                                    scores[score_name]['scale']['rel_50'][model_fold][file] = score_50
                                    scores[score_name]['scale']['rel_25'][model_fold][file] = score_25
                                    scores[score_name]['scale']['rel_10'][model_fold][file] = score_10
                                    scores[score_name]['scale']['rel_5'][model_fold][file] = score_5
                                    scores[score_name]['scale']['rel_1'][model_fold][file] = score_1
                                    scores[score_name]['scale']['exact'][model_fold][file] = score_exact
                            else:
                                for score_name in scores_to_use:
                                    above_50, above_67, above_75, above_90, total_above_50, total_above_67, total_above_75, \
                                        total_above_90 = get_results(file_path=f'./results/{model_fold}/{file}',
                                                                     metric_type=f'{score_name}_score')
                                    scores[score_name]['binary']['above_50'][model_fold][file] = above_50
                                    scores[score_name]['binary']['above_67'][model_fold][file] = above_67
                                    scores[score_name]['binary']['above_75'][model_fold][file] = above_75
                                    scores[score_name]['binary']['above_90'][model_fold][file] = above_90
                                    scores[score_name]['binary']['total_above_50'][model_fold][file] = total_above_50
                                    scores[score_name]['binary']['total_above_67'][model_fold][file] = total_above_67
                                    scores[score_name]['binary']['total_above_75'][model_fold][file] = total_above_75
                                    scores[score_name]['binary']['total_above_90'][model_fold][file] = total_above_90
            for score_name in scores:
                for score_type in scores[score_name]:
                    for score_acc in scores[score_name][score_type]:
                        print(f"\n{COLOR_YELLOW}{score_type} {score_name} {score_acc}:{COLOR_END}")
                        print_scores_as_table(scores[score_name][score_type][score_acc])
        # get sizes
        elif task == '4':
            model_smallest = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
            model_best_quality = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            model_BERT = BertModel.from_pretrained('bert-base-uncased')
            model_GloVe = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
            model_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
            model_USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            sentence_transformer_models = {"smallest": model_smallest, "best_quality": model_best_quality, "GloVe": model_GloVe, "T5": model_t5}
            # [f for f in os.listdir(folder_path)]
            for model_name, st_model in sentence_transformer_models.items():
                # Get the path to the model directory
                if model_name == 'GloVe':
                    model_directory = "\\".join(model_smallest.tokenizer.name_or_path.split("\\")[:-2]) + '\\sentence-transformers_average_word_embeddings_glove.6B.300d'
                else:
                    model_directory = st_model.tokenizer.name_or_path

                # Get the size of the model directory
                model_size_bytes = sum(
                    os.path.getsize(os.path.join(model_directory, f)) for f in os.listdir(model_directory))

                # Convert to megabytes for readability
                model_size_mb = model_size_bytes / (1024 ** 2)
                print(f"Size of '{model_name}': {model_size_mb:.2f} MB")
            model_path = hub.resolve("https://tfhub.dev/google/universal-sentence-encoder/4")
            model_size_bytes = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path))
            model_size_mb = model_size_bytes / (1024 ** 2)
            print(f"Size of 'model_USE': {model_size_mb:.2f} MB")
            model_name = model_BERT.config._name_or_path
            tokenizer = BertTokenizer.from_pretrained(model_name)
            cache_directory = tokenizer.save_pretrained(model_name)
            model_size_bytes = sum(os.path.getsize(os.path.join(model_name, f)) for f in os.listdir(model_name))
            model_size_mb = model_size_bytes / (1024 ** 2)
            print(f"Size of 'BERT': {model_size_mb:.2f} MB")
            shutil.rmtree(model_name)
        # info about dataset
        elif task == '5':
            files = os.listdir('./valid_datasets')
            for file in files:
                print(file)
                get_dataset_info(f"./valid_datasets/{file}")
            # get_dataset_info(f"./datasets/home_depot/train.csv", save_pic=False)

        if task == '6':
            print(f"{COLOR_RED}You probably didn't mean to come here!{COLOR_END}")
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/home_depot/train.csv"],
            #     new_dataset_filename="./valid_equal_datasets/home_depot.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/messages/messageWithFeature.csv"],
            #     new_dataset_filename="./valid_equal_datasets/messages.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/qnli/qnli_validation.csv"],
            #     new_dataset_filename="./valid_equal_datasets/qnli.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/quora_question_pairs/quora_duplicate_questions.tsv"],
            #     new_dataset_filename="./valid_equal_datasets/quora_question_pairs.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/rte/rte_train.csv"],
            #     new_dataset_filename="./valid_equal_datasets/rte.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/sem_sim_phones/train - train.csv"],
            #     new_dataset_filename="./valid_equal_datasets/sem_sim_phones.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/stsbenchmark/sts-train.csv"],
            #     new_dataset_filename="./valid_equal_datasets/stsbenchmark.csv"
            # )
            # dataset_preprocessing(
            #     dataset_filenames=["./datasets/wnli/wnli_train.csv"],
            #     new_dataset_filename="./valid_equal_datasets/wnli.csv"
            # )
        task = input(choices_string)
    if task == '-1':
        print("Thank you!\nGoodbye! :)")
