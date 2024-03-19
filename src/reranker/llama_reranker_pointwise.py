# -*- coding: utf-8 -*-
import json
import time
import torch
import random
import pickle
import pandas as pd
import pyterrier as pt
import ir_datasets
from typing import *
from tqdm import tqdm
from torch.nn.functional import softmax

from collections import defaultdict
from src.prompt.prompt_manager import PointwisePrompt, PromptManager
from pyterrier import Transformer
from pyterrier.model import add_ranks


if not pt.started():
    pt.init()

torch.cuda.empty_cache()

random.seed(42)

class LlamaRerankerPointwise(Transformer):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 batch_size : int = 1, 
                 k = 0,
                 other_prompt = None, 
                 is_random = False,
                 res_path = 'result'):
        
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 32000 # This stops nans for some reason, I trust github issues
        self.TRUE = self.tokenizer.encode("true")[1] # lower case to suit the prompt
        self.FALSE = self.tokenizer.encode("false")[1] # ^^
        self.batch_size = batch_size
        self.other_prompt = other_prompt
        self.res_path = res_path
        
        self._is_random = is_random
        self._k= k
   
        self.query_lookup = None
        self.docstore = None
        self.idx = 0
        self._examples = self.load_examples() if k > 0 else None
        
    def split_data_by_batch(self, prompts: List[str]):
        batch_data_list = []
        for idx in range(0, len(prompts), self.batch_size):
            batch_data_list.append(prompts[idx: idx+self.batch_size])

        return batch_data_list

    def score(self, prompts: List[str], documents: List[str] = None):
        print("start model predict...")
        outputs = []
        with torch.no_grad():
            cnt = 0
            for batch in tqdm(self.split_data_by_batch(prompts)): # Save a few LOC just directly iterating

                toks = self.tokenizer(batch, padding="longest", return_tensors="pt", add_special_tokens=True).to(self.model.device) # Same as before
                #print(toks)
                logits = self.model(**toks).logits                
                logits_after_softmax = logits[:, -1, (self.TRUE, self.FALSE)].log_softmax(dim=1) # Log softmax is generally used, can't remember why
                outputs.extend(logits_after_softmax[:, 0].cpu().detach().tolist()) # Get just the softmax prob of 'true'

        return outputs
    
    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:

        prompts = topics_or_res.apply(lambda x : PointwisePrompt(query=x.query, 
                                                                 documents=[x.text], 
                                                                 examples = self._choice_examples(x.qid) if self._k > 0 else None,
                                                                 other_prompt = self.other_prompt
                                                                ).generate(k=self._k)[0], axis=1)

        prompts = list(prompts.to_dict().values())
        scores = self.score(prompts)
        topics_or_res["score"] = scores # ordered input therefore can assign list directly
        res = add_ranks(topics_or_res)
        
        res = pd.concat([
            group.sort_values('rank') for _, group in res.groupby('qid')
        ], axis=0)

        pt.io.write_results(res, self.res_path)
        return res
    
    
        
        
    def load_examples(self):
        print(self._k)
        examples = []
        
        if self._is_random:
            '''
            print('randomly chosen training pairs')
            dataset = ir_datasets.load("msmarco-passage/train/triples-small")
            self.query_lookup = {q.query_id: q.text for q in dataset.queries_iter()}
            self.docstore = dataset.docs_store()
            #cnt = 0
            examples = []
            for docpair in dataset.docpairs_iter():
                examples.append(docpair)
                #cnt += 1
                #if cnt % 1000000 ==0 :
                #    print(cnt)
                
            '''
            print('randomly chosen training pairs')
            examples = []
            dataset = ir_datasets.load("msmarco-passage/train/triples-small")
            self.query_lookup = {q.query_id: q.text for q in dataset.queries_iter()}
            self.docstore = dataset.docs_store()

            if self._is_random:
                top = 20
                dict_file = {'19-bm25-20':845, 
                             '19-splade-20':860,
                             '20-bm25-20':1000, 
                             '20-splade-20':1000, 
                             '19-bm25-100':4205, 
                             '19-splade-100':4300,
                             '20-bm25-100':4929,
                             '20-splade-100':5000}

                names = self.res_path.split('/')[1].split('_')
                stage, k, year = names[0].lower(), names[2].split('-')[0], names[4]
                file_name = 'data/k_shot_static/'+str(dict_file[f'{str(year)}-{stage}-{str(top)}'])+'_'+ k +'.pkl'
                with open(file_name, 'rb') as file:
                    examples = pickle.load(file)
                


        else: 
            print('lexically similar examples')
            examples = {}
            with open(f"data/fewshot1920.json") as f:
                for line in f:
                    line = line.strip()
                    line = json.loads(line)
                    if 'fewshots' not in line or not line['fewshots']:
                        continue
                    fewshots = []
                    for example in line['fewshots']:
                        dic = {}
                        for key, value in example.items():
                            if key == 'msmarco.qrel.info':
                                value = value[0]
                                for k, v in value.items():
                                    if k == 'nreldoc.text':
                                        dic['neg_doc'] = v
                                    elif k == 'reldoc.text':
                                        dic['pos_doc'] = v
                                    else:
                                        dic[k] = v  
                            elif key == 'msmarco.query.text':
                                dic['query'] = value
                        fewshots.append(dic)

                    examples[line['trecdl.query.id']] = fewshots

        return examples

    
    def _choice_examples(self, query_id):
        
        if self._is_random:
            '''
            triples = self._examples
            triples = random.choices(triples, k=self._k)
            examples = []
            for triple in triples:
                dic = {}
                dic['query'] = self.query_lookup[triple.query_id]
                dic['pos_doc'] = self.docstore.get(triple.doc_id_a).text
                dic['neg_doc'] = self.docstore.get(triple.doc_id_b).text
                examples.append(dic)
            
            '''
            triples = self._examples[self.idx * self._k : self.idx * self._k + self._k]
            #triples = random.choices(triples, k=self._k)
            examples = []
            for triple in triples:
                triple = triple[0]
                dic = {}
                dic['query'] = self.query_lookup[triple.query_id]
                dic['pos_doc'] = self.docstore.get(triple.doc_id_a).text
                dic['neg_doc'] = self.docstore.get(triple.doc_id_b).text
                examples.append(dic)
            self.idx += 1
            
            
        else:
            if query_id in self._examples:
                examples = self._examples[query_id]
            else:
                examples = random.choice(list(self._examples.values()))

            examples = random.choices(examples, k=self._k)
                
        return examples
        
        
        