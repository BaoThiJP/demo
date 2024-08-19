import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from gensim.similarities import Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
import time

# Hàm tải mô hình và dữ liệu
@st.cache_data
def load_model_data():
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
    with open('gensim_similarity.pkl', 'rb') as f:
        similarity_model = pickle.load(f)
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    return tfidf_model, similarity_model, dictionary

# Tải mô hình
tfidf_model, similarity_model, dictionary = load_model_data()

# 1. Read data
hotel_info = pd.read_csv('hotel_info.csv')
# 2. Data pre-processing
# Tính số lượng giá trị không phải NaN trong từng cột
non_nan_counts = hotel_info.count()
# Lấy kiểu dữ liệu của từng cột
data_types = hotel_info.dtypes
summary_df = pd.DataFrame({
    'Non-Null Count': non_nan_counts,
    'Dtype': data_types
})
# Tính số lượng NaN
summary_df['NaN Count'] = len(hotel_info) - summary_df['Non-Null Count']
summary_df.index.name = 'Column'

#Đọc file csv sau khi cleaned,tokened
hotel_info_merge=pd.read_csv('hotel_info_merge_copy.csv')

# Đếm số lượng từ trong mỗi dòng của cột 'Hotel_Description'
hotel_info_merge['Word_Count'] = hotel_info_merge['Hotel_Description'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

# Vẽ biểu đồ histogram
plt.figure(figsize=(12, 6))
sns.histplot(hotel_info_merge['Word_Count'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Số lượng từ trong Hotel_Description')
plt.ylabel('Tần suất')
plt.title('Phân bố số lượng từ trong mô tả khách sạn')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True, axis='y', linestyle='--')
plt.show()

#4 Tạo từ điển từ vựng dictionary
# Assuming Content_gem_re contains text that needs to be tokenized
hotel_info_merge['Content_gem_re'] = hotel_info_merge['Content_gem_re'].apply(lambda x: simple_preprocess(x) if isinstance(x, str) else x)
# Obtain the number of features based on dictionary: Use corpora.Dictionary
dictionary = corpora.Dictionary(hotel_info_merge['Content_gem_re'])

# List of features in dictionary
dictionary.token2id
# Numbers of features (word) in dictionary
feature_cnt = len(dictionary.token2id)
# Obtain corpus based on dictionary (dense matrix)
corpus = [dictionary.doc2bow(text) for text in hotel_info_merge['Content_gem_re']]
# Use TF-IDF Model to process corpus, obtaining index
tfidf = models.TfidfModel(corpus)
# tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features = feature_cnt)
# Chuyển ma trận tương tự thành DataFrame
hotel_info_gensim= pd.DataFrame(index)
hotel_info_gensim


def find_similar_hotels_from_hotel_id_gensim(hotel_id, hotel_info_merge, dictionary, tfidf_model, similarity_model, num_recommendations=3):
    # Lấy chỉ số của hotel_id
    index = hotel_info_merge[hotel_info_merge['Hotel_ID'] == hotel_id].index[0]
    
    # Lấy mô tả của khách sạn
    hotel_text = hotel_info_merge.iloc[index]['Content_gem_re']
    
    # Tạo biểu diễn BoW
    hotel_bow = dictionary.doc2bow(hotel_text)
    
    # Chuyển đổi thành TF-IDF
    hotel_tfidf = tfidf_model[hotel_bow]
    
    # Tính toán độ tương đồng với tất cả các khách sạn khác
    sims = similarity_model[hotel_tfidf]
    
    # Sắp xếp kết quả theo độ tương đồng (từ cao đến thấp)
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    
    # Lấy top khách sạn có độ tương đồng cao nhất
    similar_hotels = sims[1:num_recommendations + 1]
    
    # Tạo DataFrame cho kết quả
    similar_hotels_df = pd.DataFrame([
        {
            'Hotel_ID': hotel_info_merge.iloc[hotel_index]['Hotel_ID'],
            'Hotel_Name': hotel_info_merge.iloc[hotel_index]['Hotel_Name'],
            'Hotel_Address': hotel_info_merge.iloc[hotel_index]['Hotel_Address'],
            'Similarity': similarity
        }
        for hotel_index, similarity in similar_hotels
    ])
    
    return similar_hotels_df

# Hàm hiển thị các khách sạn được đề xuất
def display_recommended_hotels(recommended_hotels, cols=3):
    num_hotels = len(recommended_hotels)
    for i in range(0, num_hotels, cols):
        current_cols = st.columns(cols)
        for j, col in enumerate(current_cols):
            if i + j < num_hotels:
                hotel = recommended_hotels.iloc[i + j]
                with col:
                    st.write(hotel['Hotel_Name'])
                    expander = st.expander(f"Description")
                    hotel_description = hotel['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")
# Lấy 10 khách sạn đầu tiên để tạo dropdown
random_hotels = hotel_info_merge.head(n=10)
# Lưu dữ liệu vào session_state
st.session_state.random_hotels = random_hotels



#GUI----------

# Hiển thị 5 hàng đầu tiên
st.title("Xây dựng mô hình")
st.write("## Conten-Based Filtering")
st.markdown("**1. Đọc dữ liệu : hotel_info.csv**")
st.dataframe(hotel_info.head())


# Hiển thị thông tin về các cột
st.markdown("**2.Thông tin dữ liệu khách sạn**")
st.dataframe(summary_df)

st.markdown("**3. Các bước xử lý dữ liệu:**")
st.markdown("""
- *Xử lý dữ liệu để phân loại thông tin theo ngôn ngữ.*
- *Sử dụng công cụ dịch google translate để chuyển đổi nội dung mô tả sang tiếng Việt.*
- *Kết hợp các bảng dữ liệu để tạo thành một tập dữ liệu thống nhất.*
- *Chuyển các giá trị "No information","#NAME?" thành NaN trong các cột "Hotel_Rank" và "Total Score".*
- *"Hotel_Rank" tách giá trị '5 sao trên 5' thành định dạng '5_sao'.*
- *Cột "Address" tách lấy thông tin phường ví dụ: 'Lộc Thọ', 'Vĩnh Hải', 'Vĩnh Phước', 'Cam Hải Đông',...*
""")

# st.code("""
#     def is_vietnamese(text):
#         try:
#             if detect(text) == 'vi':
#                 return True
#             else:
#                 return False
#         except:
#             return False
# """)

# Hiển thị biểu đồ histogram
st.write("Phân bố số lượng từ trong mô tả khách sạn")
st.pyplot(plt)



st.markdown("**4. Các bước xử lý Gensim:**")

#Hiển thị dữ liệu sau khi cleaned,tokened
st.write("Dữ liệu sau khi làm sạch và token")
selected_columns_1= ['Hotel_ID', 'Hotel_Name', 'Content_combined','Content_gem_re']
st.dataframe(hotel_info_merge[selected_columns_1].head())


st.markdown("""         
- *Tạo từ điển và ma trận thưa từ dữ liệu*
- *Áp dụng mô hình TF-IDF và xây dựng ma trận tương tự.*
""")

# # Hiển thị ma trận tương tự
# st.write("Ma trận tương tự dựa trên TF-IDF")
# st.dataframe(hotel_info_gensim)

# Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
if 'selected_hotel_id' not in st.session_state:
    st.session_state.selected_hotel_id = None

# Tạo một tuple cho mỗi khách sạn, với phần tử đầu là tên và phần tử thứ hai là ID
hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]

# Tạo dropdown cho người dùng chọn khách sạn
selected_hotel = st.selectbox(
    "Chọn khách sạn",
    options=hotel_options,
    format_func=lambda x: x[0]  # Hiển thị tên khách sạn
)

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_hotel_id = selected_hotel[1]

# Hiển thị thông tin khách sạn được chọn
if st.session_state.selected_hotel_id:
    st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
    selected_hotel = hotel_info_merge[hotel_info_merge['Hotel_ID'] == st.session_state.selected_hotel_id]

    if not selected_hotel.empty:
        st.write('#### Bạn vừa chọn:')
        st.write('### ', selected_hotel['Hotel_Name'].values[0])

        hotel_description = selected_hotel['Hotel_Description'].values[0]
        truncated_description = ' '.join(hotel_description.split()[:100])
        st.write('##### Thông tin:')
        st.write(truncated_description, '...')

        # Hiển thị danh sách các khách sạn tương tự
        st.write('#### Các khách sạn tương tự:')
        similar_hotels_df = find_similar_hotels_from_hotel_id_gensim(
            st.session_state.selected_hotel_id,
            hotel_info_merge,
            dictionary,
            tfidf_model,
            similarity_model
        )
        st.write(similar_hotels_df)

