# -*- coding: utf-8 -*-
import json
import random
import ir_datasets
import pandas as pd
import pyterrier as pt
from typing import Dict
from tqdm import tqdm

def remove_punctuation_v2(text):
    """删除标点符号"""
    for char in text:
        if char in [".", ",", ":", ";", "?", "!", "'", "/"]:
            text = text.replace(char, '')
    return text

def construct_topics(year = '19'):
    """
    构造查询集
    :param topic_file_path: 查询集地址
    :return: pd.DataFrame["qid", "query"]
    """
    ids, datas = [], []

    dataset = ir_datasets.load(f"msmarco-passage/trec-dl-20{year}/judged")
    for idx, query in enumerate(dataset.queries_iter()):
        qid = query.query_id
        query = remove_punctuation_v2(query.text)
        ids.append(idx)
        datas.append([qid, query])
    
    datas = datas[:]
    ids = ids[:]
    topics = pd.DataFrame(
        data=datas,
        index=ids,
        columns=["qid", "query"]
    )

    return topics


def construct_qrels(year = '19'):
    """
    构造qrels数据集
    :param qrels_file_path: qrels文件地址
    :return: pd.DataFrame["qid", "docno", "label"]
    """
    ids, datas = [], []
    
    dataset = ir_datasets.load(f"msmarco-passage/trec-dl-20{year}/judged")
    for idx, qrel in enumerate(dataset.qrels_iter()):
        qid = qrel.query_id
        doc_id = qrel.doc_id
        label = qrel.relevance

        ids.append(idx)
        datas.append([qid, doc_id, int(label)])

    qrels = pd.DataFrame(
        data=datas,
        index=ids,
        columns=["qid", "docno", "label"]
    )

    return qrels

