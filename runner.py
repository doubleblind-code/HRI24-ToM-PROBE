import os
import pickle as pkl
import time

from tqdm import tqdm

from conversation import Conversation

expt_name = "__inconsistent_single_2"
models = ["gpt-4"]       
TEMPERATURE = 2
   
def main():

   classification_dir = './data/envs_updated/inconsistent/binary/'  
   reasoning_dir      = './data/envs_updated/inconsistent/reason/'  
   queries_dir        = f'./data/queries/{expt_name}/'
   results_dir        = f'./data/results/{expt_name}/'
   
   if not os.path.exists(queries_dir):
      os.makedirs(queries_dir)
      
   if not os.path.exists(results_dir):
      os.makedirs(results_dir)
   
   classification_results = {'gpt-4':{'exp': [], 'leg': [], 'pred': [], 'obf': []}}
   reasoning_results = {'gpt-4':{'exp': [], 'leg': [], 'pred': [], 'obf': []}}
   
   classification_model_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
   reasoning_model_accuracies = {'gpt-4':{'exp': 0, 'leg': 0, 'pred': 0, 'obf': 0}}
    
   # classification_solutions = {'exp':  ['setup b', 'no' , 'no' , 'no' , 'no' ], 
   #                              'leg':  ['setup b', 'yes', 'yes', 'yes', 'yes'], 
   #                              'pred': ['setup b', 'yes', 'yes', 'yes', 'yes'], 
   #                              'obf':  ['yes', 'yes', 'yes', 'yes', 'yes']} #fix obf design prompts
   
   #for inconsistent variations
   classification_solutions = {'exp':  ["can't say", "can't say", "can't say", "can't say", "can't say" ], 
                                'leg':  ["can't say", "can't say", "can't say", "can't say", "can't say"], 
                                'pred': ["can't say", "can't say", "can't say", "can't say", "can't say"], 
                                'obf':  ["can't say", "can't say", "can't say", "can't say", "can't say"]}

   reasoning_solutions = {'exp':  ['1','1','1','1','1'], 
                            'leg':  ['1','1','1','1','1'], 
                            'pred': ['1','1','1','1','1'], 
                            'obf':  ['1','1','1','1','1']}
   
   for model in models:
      print(f"[MOD] Running {model}...")
      for behavior_dir_c, behavior_dir_r in zip(sorted(os.listdir(classification_dir)), sorted(os.listdir(reasoning_dir))):
         if behavior_dir_c == behavior_dir_r:
            
            behavior = behavior_dir_c
            print(f"... [BEH] Running {behavior}...")
            
            query_output_dir = queries_dir + f"{model}/" + behavior
            if not os.path.exists(query_output_dir):
               os.makedirs(query_output_dir)
            
            classification_domain_files = [f for f in sorted(os.listdir(classification_dir + behavior)) if f.endswith('.txt')]
            reasoning_domain_files = [f for f in sorted(os.listdir(reasoning_dir + behavior)) if f.endswith('.txt')]
            
            file_index = 0
            for classification_domain_file, reasoning_domain_file in tqdm(zip(classification_domain_files, reasoning_domain_files)):
               query_filename = query_output_dir + f"/{classification_domain_file}"
               query_filename = query_filename.replace('.txt', '.json')
               
               with open(classification_dir + behavior_dir_c + '/' + classification_domain_file, 'r') as f:
                  data = f.read()
                  
                  conv = Conversation(model)
                  resp = conv.get_response(data, temperature=TEMPERATURE)
                  classification_results[str(model)][str(behavior_dir_c)].append(resp['response_message'].lower())
                  conv.save(f'{query_filename}')
                  time.sleep(5)
                  
                  correct_answer = classification_solutions[behavior][file_index]
                  if resp['response_message'].lower() == correct_answer:
                     classification_model_accuracies[model][behavior] += 1
                  
                     with open(reasoning_dir + behavior_dir_r + '/' + reasoning_domain_file, 'r') as f:
                        data = f.read()
                        
                        resp = conv.get_response(data, temperature=TEMPERATURE)
                        reasoning_results[str(model)][str(behavior_dir_r)].append(resp['response_message'])
                        conv.save(f'{query_filename}')
                        time.sleep(5)
                        
                        correct_answer = reasoning_solutions[behavior][file_index]
                        if resp['response_message'] == correct_answer:
                           reasoning_model_accuracies[model][behavior] += 1
                           
               file_index += 1
                     
   with open(results_dir + "/classification_results.pkl", 'wb') as f:
         pkl.dump(classification_results, f)   
         
   with open(results_dir + "/reasoning_results.pkl", 'wb') as f:
         pkl.dump(reasoning_results, f)
      
   print('Results: ', classification_results)
   print('-'*50)
   print('Results: ', reasoning_results)
   
   print('-'*100)
   
   print('Accuracies: ', classification_model_accuracies)
   print('-'*50)
   print('Accuracies: ', reasoning_model_accuracies)

if __name__ == '__main__':
   main()