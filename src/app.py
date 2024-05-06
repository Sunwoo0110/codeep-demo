import streamlit as st
import time
import os
import pandas as pd

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

from model.story import Story

def main():
    selected_page = st.sidebar.selectbox("Select a page", [ "스토리 대화 페이지"])
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
    
    # if selected_page == "웹툰 검색하기 v1":
    #     search_page()
    if selected_page == "스토리 대화 페이지":
        story_page()

def story_page():
    st.title("Demo Page")
    
    st.subheader("등장 인물, 배경, 역할, 카테 고리 등을 통해 시놉시스를 생성하고, 주인공과 대화를 나눠보세요!")
    st.caption("..")
    
    if "story_messages" not in st.session_state:
        st.session_state["story_messages"] = [{"role": "assistant", "content": "안녕하세요. 웹툰에 대한 질문을 해주세요!"}]

    # for msg in st.session_state.search_messages:
    #     st.chat_message(msg["role"]).write(msg["content"])

    # chat_input_key = "search_chat_input_sunwoo"
    # # 사용자 인풋 받기  
    # if prompt := st.chat_input("웹툰을 검색해보세요", key=chat_input_key):
    #     # 사용자 입력 보여 주기
    #     st.session_state.search_messages.append({"role": "user", "content": prompt})
        
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     # 봇 대화 보여 주기
    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         full_response = ""
    #         # assistant_response = search.receive_chat(prompt)
    #         assistant_response = search_sunwoo.run(prompt, retriever_tool, pipeline_compression_retriever)
    #         print(assistant_response)
            
    #         message_placeholder.markdown(assistant_response)
    #     st.session_state.search_messages.append({"role": "assistant", "content": assistant_response})
    
if __name__ == "__main__":
    main()