import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
from models import TSEncoder
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from autogluon.tabular import TabularPredictor
import pandas as pd
from tqdm import tqdm

def split_all_data(all_data, all_patientids, patientid):
    train_data = all_data[all_patientids != patientid]
    train_labels = all_labels[all_patientids != patientid]
    test_data = all_data[all_patientids == patientid]
    test_labels = all_labels[all_patientids == patientid]
    return train_data, train_labels, test_data, test_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--dataset2', type=str, default='', help='The second dataset')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--hidden-dims', type=int, default=64, help='The hidden dimension (defaults to 64)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--train', action="store_true", help='Whether to perform training, if not, will load model pretrained on the second dataset')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    # TODO: Change the following split to leave one out cross validation
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        max_train_length=args.max_train_length
    )
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    model = TS2Vec(
            input_dims=8, # Hard coded for now
            device=device,
            **config
        )
    model_dir = 'training/' + args.dataset2 + '__' + name_with_datetime(args.run_name)
    model.load(f'{model_dir}/model.pkl')
    
    all_data = np.load(os.path.join('datasets', args.dataset, 'all_data.npy'))
    all__data = np.transpose(all_data, (0, 2, 1))
    all_labels = np.load(os.path.join('datasets', args.dataset, 'all_labels.npy'))[:,0]
    all_patientids = np.load(os.path.join('datasets', args.dataset, 'all_patientids.npy'))
    unique_patientids = np.unique(all_patientids)
    print('Shape of all_data:', all_data.shape)
    print('Shape of all_labels:', all_labels.shape)
    print('Number of patients:', len(unique_patientids))

    def autogluon_classification():
        aucpr = []
        for i in range(len(unique_patientids)):
            patientid = unique_patientids[i]
            print('Patient id:', patientid)
            train_data, train_labels, test_data, test_labels = split_all_data(all_data, all_patientids, patientid)
            train_repr_list, test_repr_list = [], []
            for j in tqdm(range(0, len(train_data), 200)):
                train_repr = model.encode(train_data[j: j + 200], encoding_window='multiscale')
                train_repr_list.append(train_repr)
            for j in tqdm(range(0, len(test_data), 200)):
                test_repr = model.encode(test_data[j: j + 200], encoding_window='multiscale')
                test_repr_list.append(test_repr)
            train_repr = np.concatenate(train_repr_list, axis=0)
            test_repr = np.concatenate(test_repr_list, axis=0)
            #train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
            #test_repr = model.encode(test_data, encoding_window='full_series' if test_labels.ndim == 1 else None)
            print(f'Example of train_repr: {train_repr[0]}')
            df_train, df_test = pd.DataFrame(train_repr), pd.DataFrame(test_repr)
            df_train['label'] = train_labels
            predictor = TabularPredictor(
            label='label',
            eval_metric="average_precision",
            sample_weight="auto_weight",
            ).fit(df_train, presets="best_quality", excluded_model_types=["GBM", "CAT", "XGB"], num_gpus=1)

            df_test['label'] = test_labels
            perf = predictor.evaluate_predictions(
                y_true=df_test['label'], 
                y_pred=predictor.predict_proba(df_test.drop(columns=['label'])), 
                auxiliary_metrics=True
            )
            print(f'Evaluation result on {patientid}', perf)
            aucpr.append(perf['average_precision'])
            break

        print('AUPRC:', np.mean(aucpr))
        print('AUPRC std:', np.std(aucpr))

    def linear_classification():
        aucpr = []
        for i in range(len(unique_patientids)):
            patientid = unique_patientids[i]
            print('Patient id:', patientid)
            train_data = all_data[all_patientids != patientid]
            train_labels = all_labels[all_patientids != patientid]
            test_data = all_data[all_patientids == patientid]
            test_labels = all_labels[all_patientids == patientid]
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear')
            print('Evaluation result:', eval_res)
            aucpr.append(eval_res['AUPRC'])
        print('AUPRC:', np.mean(aucpr))
        print('AUPRC std:', np.std(aucpr))
        print('AUPRC list:', aucpr)

    linear_classification()

