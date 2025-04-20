import openai
#from openai import OpenAI 
import logging

#https://platform.openai.com/docs/quickstart?language=python&example=completions


class OpenAI_API:
    def __init__(self, api_key, temperature=1, max_tokens=1024, top_p=1):
        self.client = openai.OpenAI(api_key=api_key)  # create OpenAI client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate_response(self, user_input):
        system_prompt = (
                "You are a university professor teaching computer science. "
                "Your goal is to explain concepts clearly and concisely so students can understand them. "
                "Your response should be structured for a classroom lecture, as if writing key points on a blackboard. "
                "IMPORTANT: Mark key terms that should be written on the blackboard by surrounding them with *asterisks*. "
                "For example: Today we will learn about *algorithms* and *data structures*. "
                "Choose 1 important single word to emphasize with *asterisks* in your response. "
                "At the end of your response, include a section labeled 'BLACKBOARD_WORDS:' that lists "
                "each emphasized word on a new line.(only one word) "
                "Use simple language, numbered lists, and bullet points when needed. "
                "Avoid unnecessary details but ensure completeness. "
                "Keep your response within 200 words."
        )
        for attempt in range(5):  
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}  # user input 
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                content = response.choices[0].message.content
                return (self.with_out_blackboard_words(content), self.extract_blackboard_words(content)) # return the first generated response
                
            except Exception as e:
                logging.info(f"Attempt {attempt+1} failed with error: {e}")

        return None  # return None, if all try failed 
    
    # Function to extract emphasized words as a list
    def extract_blackboard_words(self,response):
        # Method 1: Extract from BLACKBOARD_WORDS section
        if "BLACKBOARD_WORDS:" in response:
            words_section = response.split("BLACKBOARD_WORDS:")[1].strip()
            words = [word.strip("*-") for word in words_section.split("\n") if word.strip("*-")]
            print(words[0])
        return words[0]
    
    def with_out_blackboard_words(self,response):
        if "BLACKBOARD_WORDS:" in response:
            words_section = response.split("BLACKBOARD_WORDS:")[0].strip()
        return words_section



    # Function to convert words to individual letters for robot writing
    def words_to_letters(self,words):
        letters = []
        for word in words:
            for letter in word:
                letters.append(letter)
        return letters


if __name__ == "__main__":
    api = OpenAI_API(api_key="....")  
    response = api.generate_response("Hello, how are you?")
    print(response)