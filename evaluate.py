import yaml
import argparse
import pandas as pd
from evaluation.reranker_eval import pointwise_evaluation

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_experiment(experiment_config, common_config):

    model_names = common_config.get('model_names', [])
    years = common_config.get('years', [])
    first_stage_results = common_config.get('first_stage_results', [])
    topks = common_config.get('topk', [])
    
    ks = experiment_config.get('ks', [0])  
    is_random_opts = experiment_config.get('is_random_opts', [False])  
    other_prompts = experiment_config.get('other_prompts', [None])  
    
    for topk in topks:
        for model_name in model_names:
            for year in years:
                for first_stage in first_stage_results:
                    for k in ks:
                        for is_random in is_random_opts:
                            for other_prompt in other_prompts:
                                result = pointwise_evaluation(k=k, 
                                                              model_name=model_name, 
                                                              is_random=is_random, 
                                                              first_stage=first_stage, 
                                                              year=year, 
                                                              topk=topk,
                                                              other_prompt=other_prompt)
                                print(result)
                                result.to_csv('result.txt', mode='a', sep=',', float_format='%.6f', index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file', default = 'config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    common_config = config.get('common', {})
    
    for experiment_name, experiment_config in config.items():
        if experiment_name != 'common':
            print(experiment_name)
            run_experiment(experiment_config, common_config)

main()