import os
import openai
import pickle as pkl 

openai.api_key = None #insert your key here

if openai.api_key is None:
    raise Exception("Please insert your OpenAI API key in conversation.py")

class Conversation:
    def __init__(self, model) -> None:
        self.prompt = []
        self.log_history = []
        self.model =  model

    def construct_message(self, prompt, role, use_context):
        assert role in ["user", "assistant"]

        new_message = {"role": role, "content": prompt}
        if use_context:
            message = self.prompt + [new_message]
        else:
            message = [new_message]
        return message

    def get_response(self, prompt, temperature=1, role="user", use_context=True):        
        message = self.construct_message(prompt, role, use_context)

        response = openai.ChatCompletion.create(
        model=self.model,
        messages=message,
        temperature=temperature,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        answer = {"response" : response,
                                "response_message":response['choices'][0]['message']['content']
                                }
        self.log_history.append(answer)
        self.prompt = self.prompt + [{'role': role, 'content': prompt}, {'role':'assistant', 'content': answer['response_message']}]
        return answer

    def add_prompt(self, prompt, role):
        if isinstance(prompt, openai.openai_object.OpenAIObject):
            message = {"role" : role, "content": prompt['choices'][0]['message']['content']}
        else : 
            message = {"role" : role, "content": prompt}
        self.prompt.append(message)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            object = {"prompt": self.prompt, "log_history": self.log_history, "model": self.model}
            pkl.dump(object, f)


if __name__ == "__main__":
    c = Conversation("gpt-3.5-turbo")
    PROMPT_1 = "Hello, how are you?"
    PROMPT_2 = "I've been better."

    print(f"User : {PROMPT_1}")
    x1 = c.get_response(PROMPT_1)
    print("LLM : ", x1['response_message'])
    
    print(f"User : {PROMPT_2}")
    x2 = c.get_response(PROMPT_2)
    print("LLM : ", x2['response_message'])

    print("done")