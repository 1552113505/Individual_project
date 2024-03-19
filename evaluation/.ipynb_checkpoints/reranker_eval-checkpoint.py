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
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import construct_topics, construct_qrels
from src.reranker.llama_reranker_pointwise import LlamaRerankerPointwise

import warnings

warnings.filterwarnings('ignore')


if not pt.started():
    pt.init()

torch.cuda.empty_cache()


def pointwise_evaluation(k = 0, model_name = "llama2", is_random = False, first_stage = None, year = '19', topk = 20, other_prompt = None):
    """
    评估
    :conf_file:
    :return:
    """


    topics = construct_topics(year) # 读取数据集
    qrels = construct_qrels(year) #读取数据集

  
    model_dict = {"llama2": "/llm/llama-2/llama2_7b", 
                  "zephyr":"/llm/zephyr7B", 
                  "vicuna":"/llm/vicuna7B"}
    

    first_stage = f'first_stage_results/{first_stage}.dl{year}.100'
    
    model_path = model_dict[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = "left") #加载所有模型的分词器 token 
    tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # 记载特殊字符 PAD 没啥用

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": 0})
    
    text_ref = pt.get_dataset('irds:msmarco-passage') # 候选文章的总集（text） irds包保存
    
    df_results = pd.read_csv(first_stage, 
                         sep=' ', 
                         names=['qid', 'Q0', 'docno', 'rank', 'score', 'run_name'])
    df_results = df_results.drop(columns=['Q0', 'run_name'])

    df_results = pd.concat([
        group.sort_values('score', ascending=False).head(topk) for _, group in df_results.groupby('qid')
    ], axis=0)

    df_results = df_results.reset_index(drop=True)

    df_results['qid'] = df_results['qid'].astype(str)
    df_results['docno'] = df_results['docno'].astype(str)


    stage_1 = pt.Transformer.from_df(df_results)

    name = "BM25" if "bm25" in first_stage.lower() else "SPLADE"



    # 构造结果文件的名字
    if k == 0:
        name = f"{name}_{model_name}_{k}-shot_{year}"
    else:
        icl = 'static' if is_random else 'localized'
        name = f"{name}_{model_name}_{k}-shot_{icl}_{year}"
    name += f"_top{topk}"
    if other_prompt:
        name += "_" + other_prompt
        
        
    # 做pointwise的prompt拼接和模型预测，见llama-reranker文件
    
    llama_reranker = LlamaRerankerPointwise(model=model,
                                            tokenizer=tokenizer,
                                            k = k,
                                            is_random=is_random, 
                                            other_prompt=other_prompt,
                                            res_path='results/'+name)

        

    # 获得第一阶段的结果 依据id获得文本内容 然后将id与文本映射的值传到llama——reranker中 获得模型排序后的结果赋值给pipe做评估，流水线，这是一个模版，告诉流水线的任务过程，但是过程的具体任务从135行到142行 get_text让第一阶段ID对应上topics里面的查询文本
    pipe = stage_1 >> pt.text.get_text(text_ref, 'text') >> llama_reranker   
    
    # 做评估实验 利用pyterrier 输入topic和qrels 返回指标结果
    result = pt.Experiment(
        [pipe],
        topics,
        qrels,
        names=[name],
        eval_metrics=[nDCG@10, AP(rel=2), AP@topk]
    )


    return result



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--model_name', type=str, default="llama2")
    parser.add_argument('--first_stage_results', type=str, default = None)
    parser.add_argument('--is_random', action='store_true')
    parser.add_argument('--year', type=str, default = '19')
    parser.add_argument('--topk', type=int, default = 20)
    parser.add_argument('--other_prompt', type=str, default = None)
    

    args = parser.parse_args()
    
    
    result = pointwise_evaluation(k = args.k, 
                                  is_random = args.is_random, 
                                  model_name=args.model_name, 
                                  first_stage = args.first_stage_results,
                                  year = args.year,
                                  topk = args.topk,
                                  other_prompt = args.other_prompt
                                 )
    

    print(result)
    result.to_csv('result.txt', mode='a', sep=',', float_format='%.6f', index=False, header=False)
