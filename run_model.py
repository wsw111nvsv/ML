import streamlit as st
import joblib
import numpy as np

# 加载模型
model = joblib.load('logistic_regression_model.pkl')

st.title('鸢尾花分类预测')

# 创建输入表单
sepal_length = st.number_input('花萼长度', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('花萼宽度', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('花瓣长度', min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.number_input('花瓣宽度', min_value=0.0, max_value=10.0, value=0.2)

# 当用户点击预测按钮时进行预测
if st.button('预测'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    iris_classes = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    st.write(f'预测结果: {iris_classes[prediction[0]]}')
    