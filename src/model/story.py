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

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
)

class Story():
    def __init__(self):
        self.Response = Response
    
    def get_data_from_csv(self, file_path):
        """ Get data from csv file """
        loader = CSVLoader(
            file_path=file_path,
            encoding = 'UTF-8'
        )
        data = loader.load()
        return data
    
    def get_data_from_web(self, url):
        """ Get data from web """
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    
    def get_data_from_dataframe(self, data):
        """ Get multi data  """
        # docs = []
        # for data in datas:
        #     loader = DataFrameLoader(data, page_content_column="text")
        #     data = loader.lazy_load()
        #     docs.append(data)
        loader = DataFrameLoader(data, page_content_column="text")
        data = loader.lazy_load()
        return data
    
    def get_mulit_data_from_dataframe(self, datas):
        """ Get multi data  """
        docs = []
        for data in datas:
            loader = DataFrameLoader(data, page_content_column="text")
            data = loader.lazy_load()
            docs.extend(data)
        return docs
        
    def get_text_splitter(self, docs):
        """ Split text """
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        documents = text_splitter.split_documents(docs)
        return documents
    
    def get_cached_embedder(self):
        """ Get cached embedder -> Speed up """""
        underlying_embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY)
        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        return cached_embedder
    
    def get_embeddings(self, documents, cached_embedder, collection_name="story_collection"):
        vectorstore = Chroma.from_documents(
            documents, 
            cached_embedder,
            collection_name=collection_name)
        return vectorstore
    
    def get_retriever(self, vectorstore):
        """ Create a retriever """
        retriever = vectorstore.as_retriever(
            search_type="mmr",
        )
        return retriever
    
    def get_pipeline_compression_retriever(self, retriever, embeddings):
        """ Create a pipeline of document transformers and a retriever """
        ## filters
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def get_retriever_tool(self, retriever):
        """ Create a retriever tool """
        retriever_tool = create_retriever_tool(
            retriever,
            "story-retriever",
            "Query a retriever to get information about story",
        )
        return retriever_tool
    
    def parse(self, output):
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])
        
        if name == "Response":
            return AgentFinish(return_values=inputs, log=str(function_call))
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
        )
            
    def get_agent(self, retriever_tool):
        system_message = """
        You are an AI responding to users searching for webtoons. 
        Summarize the data two lines of less and answer by changing it to your own way.
        
        title: {title}
        data: {data}
        
        You always follow these guidelines:
            -Limit responses to two lines for clarity and conciseness
            -You must answer in Koreans
            -You must start with '찾으시는 작품은 {title} 입니다.'
            -You must contains the summary of the webtoon
        """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm = ChatOpenAI(
            # model_name="gpt-3.5-turbo-1106", 
            model_name="gpt-4",
            temperature=0.7, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, self.Response])
        
        agent = (
            {
                "title": lambda x: x["title"],
                "data": lambda x: x["data"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def add_docs_to_retriever(self, retriever, docs):
        retriever.add_documents(docs, ids=None)
        return retriever
            
    def get_all_relevant_documents(self, query, retriever):
        # Get relevant documents ordered by relevance score
        docs = retriever.get_relevant_documents(query)
        return docs
    
    def get_sub_relevant_documents(self, query, vectorstore):
        sub_docs = vectorstore.similarity_search(query)
        return sub_docs[0].metadata["title"]
    
    def make_retriever(self, datas):
        cached_embedder = self.get_cached_embedder()
        
        docs = self.get_mulit_data_from_dataframe(datas)
        documents = self.get_text_splitter(docs)
        
        vectorstore = self.get_embeddings(documents, cached_embedder)
        retriever = self.get_retriever(vectorstore)
        
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(retriever, cached_embedder)
        return pipeline_compression_retriever
    
    def make_story_guide(self, query):
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = """
            You are a story maker.
            You are creating a story based on the user's input.
            User's input will contain the character, background, role, and category.
            You must create a synopsis and set the detail of the main character.
            Main characters should be two people. One is the chatbot, and the other is the user.
            So the output must contain two main characters.
            Please create a story as detailed as possible.
            The target is teanagers and young adults. So, please make it suitable for them.
            You must answer in Korean.
            User's input: {query}
        """
        
        assistant_content = """
            줄거리: 황홀한 사랑, 순수한 희망, 격렬한 열정 이 곳에서 모든 감정이 폭발한다! 꿈을 꾸는 사람들을 위한 별들의 도시 ‘라라랜드’. 재즈 피아니스트 ‘세바스찬’(라이언 고슬링)과 성공을 꿈꾸는 배우 지망생 ‘미아’(엠마 스톤). 인생에서 가장 빛나는 순간 만난 두 사람은 미완성인 서로의 무대를 만들어가기 시작한다. 로스엔젤레스를 배경으로 재즈 뮤지션을 꿈꾸는 세바스찬과 배우를 꿈꾸는 미아가 만나면서 사랑에 빠지는 이야기.
            주인공(챗봇): 세바스찬: 라라랜드의 주인공. 재즈 피아니스트로서 자신의 음악을 추구하며 노래하는 것을 좋아한다.
            주인공(사용자): 미아: 세바스찬의 연인. 배우를 꿈꾸며 노래하는 것을 좋아한다.
            카테고리: 로맨스, 뮤지컬, 드라마
        """
        
        response = client.chat.completions.create(
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
    
    def run(self, query, pipeline_compression_retriever):
        retriever_tool = self.get_retriever_tool(pipeline_compression_retriever)
        result = self.get_all_relevant_documents(query, pipeline_compression_retriever)
        
        if len(result) == 0:
            return "검색 결과가 없습니다."
        agent_executor = self.get_agent(retriever_tool)
        
        response = agent_executor(
            {   
                "title": result[0].metadata["title"],
                "data": result[0].page_content,
                "input": query},
            return_only_outputs=True)
        # print(response['answer'])
        return response['answer']

if __name__ == "__main__":
    story = Story()
    story_guide = story.make_story_guide("판타지 장르로 히어로들이 많이 나오면 좋겠어. 그리고 상대 주인공은 무조건 남자로 해줘. 배경은 2050년 이후이고, 카테고리는 액션, 판타지, 모험으로 해줘.")
    print(story_guide)