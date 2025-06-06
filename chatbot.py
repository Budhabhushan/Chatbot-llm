# chatbot.py
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# class Chatbot:
#     def __init__(self, model_name="microsoft/DialoGPT-medium"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.chat_history_ids = None

#     def ask(self, user_input):
#         new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
#         if self.chat_history_ids is not None:
#             bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
#         else:
#             bot_input_ids = new_input_ids

#         self.chat_history_ids = self.model.generate(
#             bot_input_ids,
#             max_length=1000,
#             pad_token_id=self.tokenizer.eos_token_id
#         )

#         response = self.tokenizer.decode(
#             self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
#             skip_special_tokens=True
#         )
#         return response

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# BlenderbotTokenizer: Converts text to model-readable format (tokens).
# BlenderbotForConditionalGeneration: The actual BlenderBot model used to generate responses.

class Chatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def ask(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt")
        reply_ids = self.model.generate(**inputs) # High Level method for text generation
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return response.strip()