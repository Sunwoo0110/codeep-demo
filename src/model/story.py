from langchain.storage import InMemoryStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages

from langchain_community.document_transformers import  EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma,  Qdrant, FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain.text_splitter import CharacterTextSplitter

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from openai import OpenAI

import time
from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Story():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def make_story_guide(self, query):
        
        prompt = f"""
            You are a story maker.
            You are creating a story based on the user's input.
            User's input will contain the character, background, role, and category.
            You must create a synopsis that contain the ending and set the detail of the main character.
            Main characters should be the one person that is the user.
            You must create a story that contains the ending and the details of the main character.
            Please create a story as detailed as possible.
            You must refer to the famous movie or drama for the story guide.
            The target is teanagers and young adults. So, please make it suitable for them.
            You must answer in Korean.
            User's input: {query}
        """
        
        assistant_content = """
            줄거리: 황홀한 사랑, 순수한 희망, 격렬한 열정 이 곳에서 모든 감정이 폭발한다! 꿈을 꾸는 사람들을 위한 별들의 도시 ‘라라랜드’. 재즈 피아니스트 ‘세바스찬’(라이언 고슬링)과 성공을 꿈꾸는 배우 지망생 ‘미아’(엠마 스톤). 인생에서 가장 빛나는 순간 만난 두 사람은 미완성인 서로의 무대를 만들어가기 시작한다. 로스엔젤레스를 배경으로 재즈 뮤지션을 꿈꾸는 세바스찬과 배우를 꿈꾸는 미아가 만나면서 사랑에 빠지는 이야기.
            주인공(사용자): 미아:라라랜드의 주인공. 재즈 피아니스트로서 자신의 음악을 추구하며 노래하는 것을 좋아한다.
            결말: 세바스찬과 미아는 결별하며, 서로의 꿈을 이루기 위해 노력하며 서로를 응원한다. 결국 미아는 성공한 배우로, 세바스찬은 재즈 피아니스트로 자신의 꿈을 이루게 된다.
            카테고리: 로맨스, 뮤지컬, 드라마
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query },
                {"role": "assistant", "content": assistant_content}
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.8,
            frequency_penalty=0.4,
            presence_penalty=0.8,
        )
        
        return response.choices[0].message.content
    
    def make_conversation(self, query, chat_history, story_guide):
        # print(chat_history)
        # print(len(chat_history))
        
        prompt = f"""
            You are a story maker that making the multiple choices of the story based on the user's input and story guide.
            You are a chatbot that making the story based on the user's input.
            You must create the situation of some conflict and multiple choices(2 or 3) of the story based on the user's input and chat history.
            Chat history will contain the previous conversation between the user and the chatbot.
            You must create the story line based on the user's input and chat history.
            The story should be interesting and engaging.
            You must not make the story similar to the chat history. It should be unique.
            If the chat history length is more than 5, you must return the ending of the story and stop the conversation.
            If you end the conversation, you must return the ending of the story and Do not make the choices.
            You must answer as concisely as possible.
            Use your creativity to make the story as interesting as possible.
            You must not make the story and choices similar to the chat history. It should be unique.
            Please make the story very interesting and engaging. Use your creativity to make the story as interesting as possible.
            The target is teenagers and young adults. So, please make it suitable for them.
            Story must follow the ending for the story guide.
            You must answer in Korean.
            
            User's input: {query}
            Story Guide: {story_guide}
            Chat History: {chat_history}
            Chat History Length: {len(chat_history)}
        """
        
        assistant_content = """
            상황: 세바스찬은 미아에게 자신의 꿈을 이루기 위해 떠나야 한다. 
            1. 미아는 본인의 꿈을 버리고 세바스찬을 따라가기로 한다.
            2. 미아는 세바스찬와 헤어지고 자신의 꿈을 이루기로 한다.
            3. 미아와 세바스찬은 헤어지지 않고 각자의 꿈을 이루기로 한다.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query },
                {"role": "assistant", "content": assistant_content}
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.8,
            frequency_penalty=0.4,
            presence_penalty=0.8,
        )
        
        return response.choices[0].message.content
    
    def make_english_problem(self, query, chat_history, story_guide):
        # print(chat_history)
        # print(len(chat_history))
        
        prompt = f"""
            You are a english teacher that making a english problem based on the user's input and story guide.
            You are a chatbot that making the english problem based on the user's input.
            You must make a english problem that help the user to improve their english skill.
            The users are teenagers and young adults. So, please make it suitable for them.
            You must create the english problem based on the user's input and chat history.
            The english problem should be helpful for the grammar, vocabulary, and reading comprehension.
            You must refer to the famous english problem for the teenagers and young adults.
            You should find some english conversation like 'Modern Family' or 'Friends' and change the conversation that suitable for user's input and story guide.
            Please make the multiple english problems.
            
            User's input: {query}
            Story Guide: {story_guide}
            Chat History: {chat_history}
            Chat History Length: {len(chat_history)}
        """
        
        assistant_content = """
            학생 답변: 마법사에게 자신의 고민을 털어놓고 도움을 청한다.
            1. You have to tell the wizard about your worries and ask for help. What is proper sentence for this situation?
            2. What is '도움을 청하다' in English?
            3. What is '고민' in English? And make a simple sentence with '고민'.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query },
                {"role": "assistant", "content": assistant_content}
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.8,
            frequency_penalty=0.4,
            presence_penalty=0.8,
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    story = Story()
    # story_guide = story.make_story_guide("판타지 장르로 히어로들이 많이 나오면 좋겠어. 그리고 상대 주인공은 무조건 남자로 해줘. 배경은 2050년 이후이고, 카테고리는 액션, 판타지, 모험으로 해줘.")
    # print(story_guide)
    
    # tmp = """
    # 줄거리: 2050년, 판타지와 현실이 공존하는 세상. 미래 도시에서는 슈퍼히어로들이 일반인들의 평화를 지키고 있었다. 하지만 어느 날 갑자기 나타난 악당 그룹에 의해 세상은 큰 위협을 받게 된다. 주인공인 사용자는 이를 해결하기 위해 다른 히어로들과 함께 싸우게 되며, 그 과정에서 자신만의 슈퍼 파워를 발견하게 된다.

    # 주인공(사용자): 소년으로 시작한 주인공은 모험을 통해 성장하며 진짜 히어로가 되어간다. 처음에는 남들과 다르다는 이유로 스스로를 외롭게 여기던 주인공이지만, 그 차이가 바로 자신만의 특별함임을 깨닫고 강력한 슈퍼 파워를 개발한다.

    # 결말: 주인공은 마침내 악당 그룹을 물리치고 세상의 평화를 지키는데 성공한다. 그리고 그는 더 이상 자신을 외롭게 여기지 않으며, 오히려 자신의 차이를 자랑스럽게 여긴다. 그는 이제 모두에게 인정받은 히어로가 되어, 세상을 지키는 데 큰 역할을 한다.

    # 카테고리: 액션, 판타지, 모험
    # """
    
    # chat_history = {}
    # query = "내용을 시작할게"
    
    # for i in range(5):
    #     conv = story.make_conversation(query, chat_history, tmp)
    #     print("User's input: ", query)
    #     print("Story: ", conv)
    #     chat_history[i] = {
    #         "user's input": query,
    #         "story": conv
    #     }
    #     query = conv.split("\n")[4] # 임의로 선택
    chat_history = [{'role': 'assistant', 'content': '안녕하세요. 대화를 통해 스토리를 만들어보세요!'}, {'role': 'user', 'content': '안녕'}, {'role': 'assistant', 'content': '            상황: 평범한 학교생활을 보내던 리아가 어느날 갑자기 히어로의 힘을 얻게 되었습니다. 그런데 이 힘은 악에게 타겟이 될 수 있는 위험한 힘이었습니다.\n\n            1. 리아는 자신의 새로운 힘을 숨기고 평범한 생활을 계속하기로 결정합니다.\n            2. 리아는 자신의 힘을 공개하고, 세상을 구하려는 결심을 합니다.\n            3. 리아는 자신의 힘에 대해 두려워하며, 도움을 청할 사람을 찾습니다.'}, {'role': 'user', 'content': '리아는 자신의 힘에 대해 두려워하며, 도움을 청할 사람을 찾습니다.'}, {'role': 'assistant', 'content': '상황: 리아는 두려움을 이기지 못해 학교의 상담선생님인 제이에게 자신의 힘에 대해 말하게 되었습니다. 그런데 제이는 사실 리아와 같은 특별한 능력을 가진 사람들을 돕는 비밀 조직의 일원이었습니다.\n\n1. 리아는 제이와 함께 비밀 조직에 들어가 스스로를 통제하는 법을 배우기로 결정합니다.\n2. 리아는 비밀 조직에 가입하는 것을 거부하고, 다른 방법으로 자신의 힘을 제어하기로 결정합니다.\n3. 리아는 자신의 힘에 대한 두려움과 혼란 속에서 아무런 결정도 내리지 못합니다.'}, {'role': 'user', 'content': '리아는 제이와 함께 비밀 조직에 들어가 스스로를 통제하는 법을 배우기로 결정합니다.'}]
    story_guide = '스토리: "안녕"하며 인사하는 제나의 평범한 일상은 어느 날 갑자기 변합니다. 우연히 깨어난 그녀의 초능력으로 인해, 제나는 히어로가 되어버립니다. 이에 대한 부담감을 느끼지만, 주변 다른 히어로들과 친구가 되면서 점차 자신의 능력을 받아들이게 됩니다.\n\n상황: 어느 날, 제나의 마을을 파괴하려는 거대한 적이 나타납니다. \n\n선택지:\n1. 제나는 혼자서 적에 맞서 싸우려고 합니다.\n2. 제나는 다른 히어로들과 함께 적에 맞서 싸우려고 합니다.\n3. 제나는 도망치려고 합니다.'
    
    english_problem = story.make_english_problem("제나는 다른 히어로들과 함께 적에 맞서 싸우려고 합니다.", chat_history, story_guide)
    print(english_problem)
        