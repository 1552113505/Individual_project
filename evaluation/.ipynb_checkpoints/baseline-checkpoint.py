# -*- coding: utf-8 -*-
import sys
import torch
import argparse
import pandas as pd
import ir_datasets
import configparser
import transformers
import pyterrier as pt
from ir_measures import nDCG, AP
from utils.data_utils import construct_topics, construct_qrels

import warnings

warnings.filterwarnings('ignore')


if not pt.started():
    pt.init()



def baseline_evaluation(first_stage = None, year = '19'):
    """
    评估
    :conf_file:
    :return:
    """
    topics = construct_topics(year) # 读取数据集
    qrels = construct_qrels(year) #读取数据集

    first_stage = f'first_stage_results/{first_stage}.dl{year}.100'
    

    text_ref = pt.get_dataset('irds:msmarco-passage') # 候选文章的总集（text） irds包保存
    
    df_results = pd.read_csv(first_stage, 
                         sep=' ', 
                         names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'])
    df_results = df_results.drop(columns=['Q0', 'run_name'])

    df_results = pd.concat([
        group.sort_values('score', ascending=False).head(100) for _, group in df_results.groupby('qid')
    ], axis=0)

    df_results = df_results.reset_index(drop=True)

    df_results['qid'] = df_results['qid'].astype(str)
    df_results['docno'] = df_results['docno'].astype(str)

    #print(df_results)

    stage_1 = pt.Transformer.from_df(df_results)

    name = "BM25" if "bm25" in first_stage.lower() else "SPLADE"

    pipe = stage_1 >> pt.text.get_text(text_ref, 'text')

    # 做评估实验 利用pyterrier 输入topic和qrels 返回指标结果
    result = pt.Experiment(
        [pipe],
        topics,
        qrels,
        names=[name],
        eval_metrics=[nDCG@10, AP(rel=2),AP@100]
    )

    return result



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--first_stage_results', type=str, default = None)
    parser.add_argument('--year', type=str, default = '19')   

    args = parser.parse_args()
    
    result = baseline_evaluation(first_stage = args.first_stage_results,
                                  year = args.year,
                                 )
    
    print(result)
    result.to_csv('baseline.txt', mode='a', sep=',', index=False, header=False)
        

