import yaml
import argparse
import pandas as pd
from evaluation.baseline import baseline_evaluation



def run_experiment(first_stage_results, years):

    for first_stage_result in first_stage_results:
        for year in years:
            
            result = baseline_evaluation(first_stage = first_stage_result, year = year)
            print(result)
            result.to_csv('baseline.txt', mode='a', sep=',', index=False, header=False)


def main():
    first_stage_results = ['bm25', 'splade']
    years = ['19', '20']
    
    run_experiment(first_stage_results, years)

main()