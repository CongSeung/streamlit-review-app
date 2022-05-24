import streamlit as st
import joblib
import numpy as np
import pandas as pd


                                       


def run_ml():
    st.subheader('문장을 입력하면, 긍정 부정 예측해줍니다 ')

    sentence = st.text_input('문장 입력')

    # 유저가 버튼을 누르면, 예측하도록 만든다.
    if st.button('예측 실행'):

        classfier = joblib.load('data/classifier.pkl')

        vec = joblib.load('data/vec.pkl')

        new_data = np.array([sentence])         # 넘파이 어레이의 구조상 데이터가 하나여도 리스트에 넣어준다.

        X_new = vec.transform(new_data)

        X_new = X_new.toarray()

        y_pred = classfier.predict(X_new)

        if y_pred[0] == 5 :
            st.text('입력하신 문장은 긍정입니다.')
        else :
            st.text('입력하신 문장은 부정입니다.')
    