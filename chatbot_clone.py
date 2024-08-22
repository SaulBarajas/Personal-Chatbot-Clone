from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
from openai import OpenAI
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
import json
from dotenv import load_dotenv
from collections import deque


class LLMInterface(ABC):
    
    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        pass

class Llama4bitLLM(LLMInterface):
    """
    A 4-bit quantized implementation of the Llama language model.

    This class provides an interface to the Llama model, optimized for memory efficiency
    using 4-bit quantization. It implements the LLMInterface for generating responses
    in an NPC dialogue system.

    Attributes:
        tokenizer: The tokenizer used for encoding and decoding text.
        model: The loaded and quantized Llama model.
        device: The device (CPU or CUDA) on which the model is running.

    Methods:
        execute: Generates a response given a list of message dictionaries.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        login(token=hf_token)
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            #attn_implementation="flash_attention_2",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def invoke(self, messages: List[Dict[str,str]]) -> str:
        """
        Generate a response using the 4-bit quantized Llama model.

        Args:
            messages (List[Dict[str,str]]): A list of message dictionaries representing the conversation history.

        Returns:
            str: The generated response from the Llama model.
        """
        # Use the model's chat template to format the input
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        
        # create attention mask
        attention_mask = torch.ones_like(inputs)
        attention_mask[inputs == self.tokenizer.pad_token_id] = 0
        
        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask= attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=1024, # max tokens generates
                do_sample=False, # the model uses sampling to generate tokens, which introduces randomness into the output. top_k should be 0.
                #temperature=0.7, # adjusts the randomness (lower values make it more deterministic)
                #top_p=0.95, # (nucleus sampling) limits the sampling to the most probable tokens that cumulatively add up to the set probability mass
                #top_k=0, # limits the sampling pool to only the k most likely next tokens.
                #beam_search=2, # At time step 1, besides the most likely hypothesis ("The","nice"), beam search also keeps track of the second most likely one ("The","dog"). 
                                # At time step 2, beam search finds that the word sequence ("The","dog","has"), has with 0.36 a higher probability than ("The","nice","woman"), which has 0.2
                #num_return_sequences=5, # Generates multiple sequences for beam search
                #no_repeat_ngram_size=2, # Prevents the model from repeating the same n-gram within a sequence (word sequences)
                #early_stopping=True, #so that generation is finished when all beam hypotheses reached the EOS token.
            )
        
        # Decode and return the generated response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("assistant\n")[-1].strip()
        
        return assistant_response
    
class llamaLLM(LLMInterface):
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), 
                    base_url="https://openrouter.ai/api/v1",)
    
    def invoke(self, messages: List[Dict[str,str]]) -> str:
        completion = self.client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-3.1-8b-instruct",
            extra_body={
                "temperature": 0.0,
                "provider": {"order": ["Fireworks", "OctoAI"]},
            },
        )
        return completion.choices[0].message.content
    
    
class SaveableChat:
    def __init__(self, chat_id: str, user_id: str, llm: LLMInterface):
        self.chat_id = chat_id
        self.user_id = user_id
        self.llm = llm
        self.full_conversation_history = []  # For record-keeping
        self.name = None
        self.system_prompt = "You are a helpful assistant."
        self.recent_messages = deque(maxlen=500)  # Adjust maxlen as needed
        self.running_summary = ""
        self.message_count = 0

        
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        
    def set_name(self, name: str):
        self.name = name
        
    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content}
        
        # Capture the message that will be dropped
        dropped_message = None
        if len(self.recent_messages) == self.recent_messages.maxlen:
            dropped_message = self.recent_messages[0]
        
        self.recent_messages.append(message)
        self.full_conversation_history.append(message)
        self.message_count += 1
        
        if dropped_message and self.message_count > self.recent_messages.maxlen:
            self.update_running_summary(dropped_message)


    
    def update_running_summary(self, dropped_message):

        summary_prompt = f"""
        Please update the following summary with the new information:

        Current summary: {self.running_summary}

        Dropped message to be summarized:
        {dropped_message['role'].capitalize()}: {dropped_message['content']}

        Provide a concise updated summary.
        """

        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that updates conversation summaries. Make sure to keep the summary concise and relevant, while taking into account what the user and assistant have said."},
            {"role": "user", "content": summary_prompt}
        ]

        self.running_summary = self.llm.invoke(summary_messages)


    def get_response(self) -> str:
        full_conversation = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if self.running_summary:
            full_conversation.append({"role": "system", "content": f"Previous conversation summary: {self.running_summary}"})
        
        full_conversation.extend(list(self.recent_messages))

        response = self.llm.invoke(full_conversation)
        self.add_message("assistant", response)
        return response
    
    def get_full_history(self) -> List[Dict[str, str]]:
        return self.full_conversation_history

    def save(self, directory: str = "saved_chats"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.user_id}_{self.chat_id}.json"
        with open(f"{directory}/{filename}", "w") as f:
            json.dump({
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "name": self.name,
                "recent_messages": list(self.recent_messages),
                "running_summary": self.running_summary,
                "system_prompt": self.system_prompt,
                "message_count": self.message_count,
                "full_conversation_history": self.full_conversation_history
            }, f)
            
    @classmethod
    def load(cls, chat_id: str, user_id: str, llm: LLMInterface, directory: str = "saved_chats"):
        chat = cls(chat_id, user_id, llm)
        filename = f"{user_id}_{chat_id}.json"
        try:
            with open(f"{directory}/{filename}", "r") as f:
                data = json.load(f)
                chat.user_id = data.get("user_id", user_id)
                chat.chat_id = data.get("chat_id", chat_id)
                chat.name = data.get("name")
                chat.recent_messages = deque(data.get("recent_messages", []), maxlen=chat.recent_messages.maxlen)
                chat.running_summary = data.get("running_summary", "")
                chat.system_prompt = data.get("system_prompt", chat.system_prompt)
                chat.message_count = data.get("message_count", 0)
                chat.full_conversation_history = data.get("full_conversation_history", [])
        except FileNotFoundError:
            pass
        return chat
    
    def generate_name(self, user_input: str) -> str:
        name_prompt = f"""
        Please generate a name for a chat that starts with this message: '{user_input}'
        
        Your response should only be the name of the chat, nothing else.
        
        For example:
        User input: "Hello, how are you?"
        Chat name: "Greeting"
        
        User input: "I need help with my resume"
        Chat name: "Resume Help"
        
        User input: "How can I improve my cover letter?"
        Chat name: "Cover Letter Help"
        
        User input: "What are the best interview questions to ask?"
        Chat name: "Asking Interview Questions"
        
        User input: "How do you code a chatbot?"
        Chat name: "Coding a Chatbot"
        
        """
        naming_prompt = [{"role": "system", "content": "You are a chat naming assistant. Your task is to create a short, relevant name for a chat based on the user's first message. The name should be concise and descriptive."},
                         {"role": "user", "content": name_prompt}]
        self.name = self.llm.invoke(naming_prompt).strip()
        return self.name
    
    

class Chatbot:
    def __init__(self, llm: LLMInterface, user_id: str):
        self.llm = llm
        self.user_id = user_id
        self.chats = {}
        self.load_user_chats()

    # creates a new chat or loads an existing one
    # TODO: Add to API
    def create_chat(self, chat_id: str, user_id: str) -> SaveableChat:
        key = f"{user_id}_{chat_id}"
        if key not in self.chats:
            self.chats[key] = SaveableChat(chat_id, user_id, self.llm)
        return self.chats[key]

    # gets a chat by chat_id and user_id and returns it. Adds it to chats dictionary if it doesn't exist
    def get_chat(self, chat_id: str, user_id: str) -> SaveableChat:
        key = f"{user_id}_{chat_id}"
        if key not in self.chats:
            self.chats[key] = SaveableChat.load(chat_id, user_id, self.llm)
        return self.chats[key]
    
    # loads all chats for a user
    def load_user_chats(self, directory: str = "saved_chats") -> List[SaveableChat]:
        user_chats = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.startswith(f"{self.user_id}_") and filename.endswith(".json"):
                    chat_id = filename[len(self.user_id)+1:-5]  # Remove user_id_ prefix and .json suffix
                    chat = self.get_chat(chat_id, self.user_id)
                    user_chats.append(chat)
        return user_chats
    
    # gets a summary of all chats for a user
    # TODO: Add to API
    def get_user_chat_summaries(self) -> List[Dict[str, Union[str, int, List[Dict[str, str]]]]]:
        chats = self.load_user_chats()
        summaries = []
        for chat in chats:
            full_history = chat.get_full_history()
            summaries.append({
                "chat_id": chat.chat_id,
                "name": chat.name,
                "last_message": full_history[-1]["content"] if full_history else "",
                "full_history": full_history,
                "message_count": chat.message_count,
                "recent_messages": list(chat.recent_messages),
                "running_summary": chat.running_summary
            })
        return summaries

    # sets the system prompt for a chat
    # TODO: Add to API
    def set_system_prompt(self, user_id: str, chat_id: str, prompt: str):
        chat = self.get_chat(chat_id, user_id)
        chat.set_system_prompt(prompt)
        chat.save()  # Save the chat after updating the system prompt
        
    def set_name(self, user_id: str, chat_id: str, name: str):
        chat = self.get_chat(chat_id, user_id)
        chat.set_name(name)
        chat.save()
        
    def set_llm(self, llm: LLMInterface):
        self.llm = llm
        # Update the LLM for all existing chats
        for chat in self.chats.values():
            chat.llm = llm

    # chats with the user
    # TODO: Add to API
    def chat(self, chat_id: str, user_id: str, user_input: str) -> str:
        chat = self.get_chat(chat_id, user_id)
        if not chat.full_conversation_history:
            chat.generate_name(user_input)
        chat.add_message("user", user_input)
        response = chat.get_response()
        chat.save()
        return response
    
# TODO: Multi-user support: Implement a user management system to handle multiple users with authentication.
#          nChat summarization: Add a feature to generate summaries of long conversations. TODO: Need to test this.
# TODO: Voice input/output: Add support for voice messages or text-to-speech for responses.
# TODO: Image generation: Integrate an image generation model to create images based on text prompts.
# TODO: Context-aware responses: Implement a feature to maintain longer-term context across multiple chat sessions.
# TODO: Integration with external APIs: Add the ability to fetch real-time data (e.g., weather, news) to enhance responses.
# TODO:  File attachment support: Allow users to attach and share files within the chat.
    
if __name__ == "__main__":
    load_dotenv()
    llm = llamaLLM()
    chatbot = Chatbot(llm, "saulb423")
    chatbot.create_chat(user_id="saulb423",  chat_id="1")
    chatbot.chat(user_id="saulb423", chat_id="1", user_input="Hey can you help me prepare a sales pitch for a job interview?")
    print(chatbot.get_user_chat_summaries())
    print(len(chatbot.get_chat(chat_id="1", user_id="saulb423").full_conversation_history))
    print(len(chatbot.get_chat(chat_id="1", user_id="saulb423").recent_messages))





    