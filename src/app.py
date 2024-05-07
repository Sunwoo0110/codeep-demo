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
    model = Story()
    st.title("Demo Page")
    st.caption("등장 인물, 배경, 역할, 카테 고리 등을 통해 시놉시스를 생성하고, 주인공과 대화를 나눠보세요!")
    
    # st.subheader("등장 인물, 배경, 역할, 카테 고리 등을 통해 시놉시스를 생성하고, 주인공과 대화를 나눠보세요!")
    # st.caption("판타지 장르로 히어로들이 많이 나오면 좋겠어. 그리고 주인공은 무조건 여자로 해줘. 배경은 2050년 이후이고, 카테고리는 액션, 판타지, 모험으로 해줘.")
    
    story_prompt = st.text_area("원하는 스토리의 시놉시스를 생성 해보세요!", value="판타지 장르로 히어로들이 많이 나오면 좋겠어. 그리고 주인공은 무조건 여자로 해줘. 배경은 2050년 이후이고, 카테고리는 액션, 판타지, 모험으로 해줘.", height=30)
    
    if st.button("시놉시스 생성하기"):
        story_synopsis = model.make_story_guide(story_prompt)
        st.session_state["story_synopsis"] = story_synopsis
    
    try:
        story_guide = st.session_state["story_synopsis"]
    except KeyError:
        story_guide = ""
    st.write(story_guide)
    
    if "story_messages" not in st.session_state:
        st.session_state["story_messages"] = [{"role": "assistant", "content": "안녕하세요. 대화를 통해 스토리를 만들어보세요!"}]

    for msg in st.session_state.story_messages:
        print(msg)
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input_key = "story_chat_input_sunwoo"
    
    # 사용자 인풋 받기  
    if prompt := st.chat_input("대화를 만들어보세요", key=chat_input_key):
        # 사용자 입력 보여 주기
        st.session_state.story_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 대화 보여 주기
        with st.chat_message("assistant"):
            chat_history = st.session_state.story_messages
            message_placeholder = st.empty()
            
            story_guide = st.session_state["story_synopsis"] if st.session_state["story_synopsis"] else ""
            
            assistant_response = model.make_conversation(prompt, chat_history, story_guide)
            print(assistant_response)
            message_placeholder.markdown(assistant_response)
            
            # chat_history[len(st.session_state.story_messages)//2] = {
            #     "user's input": prompt,
            #     "story": assistant_response
            # }
        st.session_state.story_messages.append({"role": "assistant", "content": assistant_response})
    
if __name__ == "__main__":
    main()