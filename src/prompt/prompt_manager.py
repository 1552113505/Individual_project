# -*- coding: utf-8 -*-
import json
import random
from typing import *


class PromptManager(object):
    def __init__(self, query: str, documents: List[str]):
        self.query = query
        self.documents = documents
        
    def generate(self):
        raise NotImplementedError 
        

class PointwisePrompt(PromptManager):
    def __init__(self, query: str, documents: List[str], examples: List[str] = None, is_k_shots: bool = False, other_prompt: str = None):
        super().__init__(query=query, documents=documents)

        self._examples = examples
        self.other_prompt = other_prompt
        
        self.sys_prompt = " You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query." if not other_prompt and not examples else ""
        self.instruction = " If the document below aligns with the ensuing query, output ’true’, else output ’false’."
        self.explanation = " After stating your decision, explain why you made this choice." if other_prompt == "explanation" else ""
        self.k_shots = "\n{}\n" if examples else " "
        self.query_prompt = "query: [{}] document: [{}] relevant: "
        
        self.prompt_template = self.sys_prompt + self.instruction + self.explanation + self.k_shots + self.query_prompt
        
        
            
    def generate_examples(self, k: int):
        examples = []
        for example in self._examples:
            example_pos = "query: [{}], document: [{}], relevant: {}".format(example["query"], example["pos_doc"], "true")
            example_neg = "query: [{}], document: [{}], relevant: {}".format(example["query"], example["neg_doc"], "false")
            examples.extend([example_pos, example_neg])

        return examples

    def generate(self, k: int = 0):
        prompts = []
        
        if k == 0:
            for document in self.documents:
                prompts.append(self.prompt_template.format(self.query, document))
        else: 
            for document in self.documents:
                prompts.append(self.prompt_template.format("\n".join(self.generate_examples(k)), self.query, document))
        #print(prompts)
        return prompts

