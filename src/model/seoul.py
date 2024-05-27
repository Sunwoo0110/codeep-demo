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

class Seoul():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def make_seoul_story(self, query):
        
        prompt = f"""
            You are a story maker. This story will be used in a story-based interative game.
            You are creating a story based on the user's input.
            User's input will contain the character, background, role, and category.
            Most import thing is that user'input will contain the content of the famous game.
            You must create a story that based on the user's input.
            You must create a 5~10 of episode that make the game user to go through the end of story.
            The user's input will contain the several episodes, so you must refer to this. 
            You can make the same as user's input, but you must contain a different storys.
            You must create a synopsis that contain the ending.
            Please create a story as detailed as possible. 
            Please create a episode as detailed as possible.
            You must answer in Korean.
            User's input: {query}
        """
        
        assistant_content = f"""
            전체 스토리: 2015년 서울 상공에서 원인 모를 핵폭발이 일어나고 18년 후, 하나는 100명 남짓한 도봉산의 어느 생존자 공동체에서 자란 주인공의 시점으로, 가족을 죽인 범인을 찾아 나서는 여정을 따라가며, 하나는 그 동안 기억을 잃고, 자신의 기억을 찾아 나서는 게임
            결말: 비록 수많은 피를 흘렸으나 엽우회-기술자 연합은 승리했고, 더 이상의 전쟁이 없길 바라며 7경비단이 남겨놓은 막대한 병기들을 땅에 파묻은 뒤 공동 관리하기로 한다. 주인공은 7경비단의 막사가 있던 자리에 병원과 학교를 세우고 서울 사람들에게 존경받는 위인이 된다.
            에피소드:
                1. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                2. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                3. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                4. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                5. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                6. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                7. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                8. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                9. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
                10. 게임 시작 후 조금 진행하면 마을에서 찾아온 폐품업자 김씨가 돈 1칸이나 식량 하나, 혹은 둘 다 주며 마을 사람들의 안부를 전하고, 뒷산에서 발견된 군화 발자국에 대해 얘기하며 랜덤으로 서울 중심지에 자리한 술집 '마님'이나 보부상 집단 '일신상회'로 가라고 한다. 소문을 따라 7경비단을 찾아 나서면 성균관대학교에 도착하고 학생회의 투표를 거쳐 7경비단으로 정찰을 가게 된다. 진입 시에 '교섭'이나 '순수한 얼굴'을 가지고 있다면 대외협력국장을 확정적으로 획득하며, 학생회에 들어가고 싶다고 하면 서술형 시험을 보고 얻을 수 있다.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query },
                {"role": "assistant", "content": assistant_content}
            ],
            temperature=0.8,
            max_tokens=4096,
            top_p=0.8,
            frequency_penalty=0.4,
            presence_penalty=0.8,
        )
        
        return response.choices[0].message.content
    
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content


if __name__ == "__main__":
    seoul = Seoul()
    
    file_path = "data/seoul.txt"
    content = seoul.read_file(file_path)
    
    query = f"""
        서울을 기반으로 한 재난, 아포칼립스 소재의 스토리를 만들어주세요.
        content: {content}
    """
    response = seoul.make_seoul_story(query)
    print(response)