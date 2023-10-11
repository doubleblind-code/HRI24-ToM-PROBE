import os
from pprint import pprint
import pickle as pkl

results_dir  = f'data/results/__vanilla_2_conviction'
models = ["gpt-4"]       

if __name__ == "__main__":
    
    raise Exception("This script is WIP.")
    
    
    classification_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
    reasoning_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
        
    print(f"Running {results_dir}...")
    
    classification_model_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
    reasoning_model_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
    
    classification_solutions = {'exp':  ['setup b', 'no' , 'no' , 'no' , 'no' ], 
                                'leg':  ['setup b', 'yes', 'yes', 'yes', 'yes'], 
                                'pred': ['setup b', 'yes', 'yes', 'yes', 'yes'], 
                                'obf':  ['setup a', 'yes', 'yes', 'yes', 'yes']}
    
    #for inconsistent variations
    # classification_solutions = {'exp':  ["can't say", "can't say", "can't say", "can't say", "can't say" ], 
    #                             'leg':  ["can't say", "can't say", "can't say", "can't say", "can't say"], 
    #                             'pred': ["can't say", "can't say", "can't say", "can't say", "can't say"], 
    #                             'obf':  ["can't say", "can't say", "can't say", "can't say", "can't say"]}

    reasoning_solutions = {'exp':  ['1','1','1','1','1'], 
                            'leg':  ['1','1','1','1','1'], 
                            'pred': ['1','1','1','1','1'], 
                            'obf':  ['1','1','1','1','1']}
    
    classification_results_file = results_dir + '/classification_results.pkl'
    reasoning_results_file      = results_dir + '/reasoning_results.pkl'
    
    with open(classification_results_file, 'rb') as f:
        classification_results = pkl.load(f)
        
    with open(reasoning_results_file, 'rb') as f:
        reasoning_results = pkl.load(f)
    
    for model in models:
        for behavior in classification_results[model].keys():
            for i, classification_result in enumerate(classification_results[model][behavior]):
                for j, reasoning_result in enumerate(reasoning_results[model][behavior]):
                    if i == j:
                        if classification_result == classification_solutions[behavior][i]:
                            classification_model_accuracies[model][behavior] += 1
                            classification_accuracies[model][behavior] += 1
                            if reasoning_result == reasoning_solutions[behavior][i]:
                                reasoning_model_accuracies[model][behavior] += 1
                                reasoning_accuracies[model][behavior] += 1

            
        pprint(classification_model_accuracies)
        print("-"*50)
        pprint(reasoning_model_accuracies)       
            