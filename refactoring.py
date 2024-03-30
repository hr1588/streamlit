import pandas as pd
import streamlit as st
import altair as alt
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import networkx as nx
from gensim.models import Word2Vec
from gensim import corpora, models
from tqdm import tqdm
from collections import Counter
import re
import bareunpy as brn
import gensim
from gensim.models.coherencemodel import CoherenceModel
import plotly.graph_objects as go
import string

plt.rc('font', family='Malgun Gothic') 
plt.rc('axes', unicode_minus=False)

# 데이터 로딩 및 초기화
df = pd.read_csv('weight_df.csv')
df['date'] = pd.to_datetime(df['date'])

# 스트림릿 앱 설정
st.title('Sentiment Analysis')
st.sidebar.header('Filter Options')

# 사이드바 설정
sentiment_option = st.sidebar.selectbox(
    'Choose sentiment',
    ('all', 'positive', 'neutral', 'negative')
)

# 사이드바 설정: 날짜 선택 (pd.Timestamp로 변환)
start_date = pd.Timestamp(st.sidebar.date_input('Start date', value=pd.to_datetime('2024-02-28')))
end_date = pd.Timestamp(st.sidebar.date_input('End date', value=pd.to_datetime('2024-03-28')))

# 데이터 필터링
def filter_data(df, sentiment, start_date, end_date):
    if sentiment != 'all':
        df = df[df['sentiment'] == sentiment]
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

filtered_df = filter_data(df, sentiment_option, start_date, end_date)

# 데이터 시각화
def visualize_data(df):
    # 일자별 댓글 수 시각화
    comments_per_day = df.groupby(df['date'].dt.date)['content'].count().reset_index(name='comments_count')
    comments_chart = alt.Chart(comments_per_day).mark_bar().encode(
        x=alt.X('date:T',axis=alt.Axis(format='%m/%d')),
        y='comments_count:Q',
        tooltip=['date', 'comments_count']
    ).properties(title='일별 댓글 수 현황')
    st.altair_chart(comments_chart, use_container_width=True)

visualize_data(filtered_df)
sorted_df = filtered_df.sort_values(by='likes', ascending=False)
st.write('DataFrame', sorted_df.head().reset_index(drop=True))

########################### 감성 점수 분포 시각화 ################################

st.markdown('''
    ## 가중 감성 점수에 대한 설명

    감성 점수는 데이터 내 각 항목의 감성을 나타내는 지표입니다. 이 점수는 세 가지 값으로 구분됩니다: 긍정적인 경우 `1`, 중립적인 경우 `0`, 부정적인 경우 `-1`입니다. 이 점수는 각 항목의 '좋아요' 수와 결합되어 '가중 감성 점수'를 형성합니다.

    - 긍정적(`1`) 또는 부정적(`-1`) 감성을 가진 항목의 경우, 해당 항목의 '좋아요' 수를 로그화하여 감성 점수와 곱함으로써 가중치를 적용합니다.
    - 중립적(`0`) 감성을 가진 항목의 경우, 감성 점수 자체가 `0`이므로, 로그화한 '좋아요' 수를 직접 가중 감성 점수로 사용합니다.

    이 과정을 통해, 단순 감성 분석을 넘어서 사용자 반응의 강도까지 고려한 더욱 세밀한 데이터 분석이 가능해집니다. 가중 감성 점수는 각 항목이 얼마나 긍정적, 중립적, 또는 부정적인지, 그리고 해당 감성이 사용자들 사이에서 얼마나 강력하게 반영되는지를 동시에 반영합니다.
''')
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('## 감정 점수 시각화')
st.markdown('각 막대는 데이터셋 내의 감성 점수의 분포를 나타냅니다. X축은 감성 점수를, Y축은 해당 점수의 빈도수를 표시합니다. 이를 통해 가장 빈번하게 나타나는 감성 점수의 범위를 쉽게 파악할 수 있습니다.')

st.markdown('<br>', unsafe_allow_html=True)

def visualize_score_distribution(df):
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("weighted_sentiment_score:Q", bin=True, title='Sentiment Score'),
        y=alt.Y('count()', title='Frequency'),
        tooltip=[alt.Tooltip('count()', title='Frequency')]
    ).properties(title='1. 감성 점수 분포')

    text = hist.mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # 텍스트 위치 조정
    ).encode(text='count()')

    final_chart = hist + text
    st.altair_chart(final_chart, use_container_width=True)

# 시간에 따른 평균 감정 점수 계산 및 시각화
def visualize_average_score_over_time(df):
    average_sentiment_score_by_date = df.groupby('date')['weighted_sentiment_score'].mean().reset_index()

    line_chart = alt.Chart(average_sentiment_score_by_date).mark_line(point=True).encode(
        x=alt.X('date:T', axis=alt.Axis(format='%m/%d', title='Date')),
        y=alt.Y('weighted_sentiment_score:Q', title='Average Sentiment Score'),
        tooltip=[alt.Tooltip('date:T', format='%m/%d'), 'weighted_sentiment_score:Q']
    ).properties(title='2. 일자별 감성 점수 평균', width=600)

    st.altair_chart(line_chart, use_container_width=True)

# 필터링된 데이터프레임 호출
visualize_score_distribution(filtered_df)
visualize_average_score_over_time(filtered_df)

################## 결과 신뢰도 및 기간내 감성 평균점수 표 #################

st.markdown('## 결과 신뢰도 및 기간내 감성 평균점수')

# 감성 옵션에 따라 데이터 필터링
if sentiment_option != 'all':
    df = df[df['sentiment'] == sentiment_option]
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# 신뢰도와 가중 감성 점수의 평균을 계산
st.markdown('''
선택한 감성 옵션(Choose sentiment)에 따라 필터링된 데이터프레임에서 결과 신뢰도(confidence)와 기간 내 감성 평균점수(weighted_sentiment_score)를 계산하고, 이를 표로 출력합니다.
''')

st.markdown('<br>', unsafe_allow_html=True)

aggregated_data = filtered_df.groupby('sentiment', as_index=False).agg({
    'confidence': 'mean',
    'weighted_sentiment_score': 'mean'
})

# 결과 표시
st.dataframe(aggregated_data)

##################### 감성별 상위 단위 출력 #####################

st.markdown('## 감성별 상위 단어 출력')

# 불용어 로딩 함수
def load_stopwords():
    with open('stopwords-ko.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    additional_stopwords = {'던파', '게임'}  # 추가 불용어
    stopwords.extend(additional_stopwords)
    return set(stopwords)

# 텍스트 데이터를 토큰화하고, 불용어를 제거하는 함수
def tokenizer(data, api_key):
    # Tagger 인스턴스 생성 및 사용자 정의 사전 설정
    t = brn.Tagger(api_key, "localhost", 5656)
    cust_dic = t.custom_dict("my_dict_01")
    cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
    cust_dic.update()
    t.set_domain("my_dict_01")
    
    # 불용어 로딩 (이 과정은 함수 외부에서 한 번만 수행되어야 합니다)
    stopwords = load_stopwords()

    tags = t.tags(data)
    nouns = tags.nouns()

    # 불용어 제거 및 단어 길이 필터링
    filtered_nouns = [word for word in nouns if word not in stopwords and len(word) >= 2]
    
    # 단어 빈도수 계산 및 DataFrame 변환
    count_tokens = Counter(filtered_nouns)
    df_tokens = pd.DataFrame(count_tokens.items(), columns=['word', 'count'])
    df_tokens = df_tokens[df_tokens['count'] >= 10].sort_values('count', ascending=False).reset_index(drop=True)
    
    return df_tokens

# 주어진 빈도 데이터를 바탕으로 시각화하는 함수
def visualize_word_frequencies(words_df, top_n):
    # 데이터프레임을 미리 정렬
    words_df['count'] = words_df['count'].astype(int)

    chart = alt.Chart(words_df).mark_bar().encode(
        y=alt.Y('word:N', title='Word',sort=alt.EncodingSortField(field='count', order='descending')),
        x=alt.X('count:Q', title='Frequency'),
        tooltip=['word', 'count']
    ).properties(title='상위 단어 빈도 수', width=600, height=600)
    
    text = chart.mark_text(align='left', baseline='middle', dx=3).encode(text='count:Q')
    final_chart = chart + text
    st.altair_chart(final_chart, use_container_width=True)

# 실행 부분
api_key = "koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA"

if st.toggle('시각화'):
    top_n = st.number_input('표시할 단어 수:', min_value=1, value=20)
    tokens_df = tokenizer(filtered_df['content'].tolist(), api_key)
    visualize_word_frequencies(tokens_df, top_n)
    # st.write(tokens_df.sort_values('count', ascending=False).head())

########################## LDA ###############################
    
# 텍스트 전처리 및 불용어 처리
def preprocess_text(lines, tagger, stopwords):
    tokenized_lines = []
    for line in tqdm(lines):
        tagged = tagger.tags([line])
        nouns = tagged.nouns()
        tokens = [word for word in nouns if len(word) > 1 and word not in stopwords]
        tokenized_lines.append(tokens)
    return tokenized_lines

st.title('LDA Topic Modeling')

# 토픽 별 주요 단어와 비중을 시각화하는 함수
def visualize_topic_words(lda_model, num_topics):
    topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    for tid, topic in topics:
        st.write(f"토픽 {tid + 1}:")
        df_topic = pd.DataFrame(topic, columns=['단어', '비중'])
        chart = alt.Chart(df_topic).mark_bar().encode(
            x=alt.X('비중:Q', title='비중'),
            y=alt.Y('단어:N', sort='-x', title='단어'),
            color=alt.Color('단어:N', legend=None),
            tooltip=['단어', '비중']
        ).properties(title=f'토픽 {tid + 1}의 주요 단어 분포', width=600, height=300)
        st.altair_chart(chart)

# LDA 모델링 및 시각화 함수
def LDA_display(data, num_topics, width, height):
    api_key = "koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA"
    t = brn.Tagger(api_key, "localhost", 5656)
    cust_dic = t.custom_dict("my_dict_01")
    cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
    cust_dic.update()
    t.set_domain("my_dict_01")
    stopwords = load_stopwords()
    
    tokenized_lines = preprocess_text(data, t, stopwords)
    dictionary = corpora.Dictionary(tokenized_lines)
    corpus = [dictionary.doc2bow(text) for text in tokenized_lines]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=4)
    
    visualize_topic_words(lda_model, num_topics)

    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
    st.components.v1.html(pyLDAvis_html, width=width, height=height, scrolling=True)

# LDA 모델링 실행
if st.toggle('LDA'):
    LDA_display(filtered_df['content'].tolist(), 10, 1200, 700)

##### LDA 개수 탐색 #####

# def preprocess_and_create_corpus(data, api_key, stopwords):
#     t = brn.Tagger(api_key, "localhost", 5656)
#     cust_dic = t.custom_dict("my_dict_01")
#     cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
#     cust_dic.update()
#     t.set_domain("my_dict_01")
#     stopwords = load_stopwords()

#     tokenized_lines = preprocess_text(data, t, stopwords)
#     dictionary = corpora.Dictionary(tokenized_lines)
#     corpus = [dictionary.doc2bow(text) for text in tokenized_lines]
    
#     return corpus, dictionary, tokenized_lines


# def train_lda_and_find_optimal_topics(corpus, dictionary, tokenized_lines, start, limit, step):
#     coherence_values = []
#     model_list = []

#     for num_topics in range(start, limit, step):
#         model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=tokenized_lines, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
        
#     # 최적의 토픽 수 결정
#     max_coherence_val_index = coherence_values.index(max(coherence_values))
#     optimal_topics = range(start, limit, step)[max_coherence_val_index]
#     optimal_model = model_list[max_coherence_val_index]

#     return optimal_model, optimal_topics, coherence_values

# if __name__ == '__main__':
#     # 데이터 로드 및 전처리
#     data = filtered_df['content'].tolist() 
#     api_key = "koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA"
#     stopwords = load_stopwords()

#     # LDA 모델 훈련 및 최적의 토픽 수 찾기
#     start, limit, step = 2, 15, 1
#     corpus, dictionary, tokenized_lines = preprocess_and_create_corpus(data, api_key, stopwords)
#     optimal_model, optimal_topics, coherence_values = train_lda_and_find_optimal_topics(corpus, dictionary, tokenized_lines, start, limit, step)

#     # LDA 시각화
#     width, height = 1200, 700
#     vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus, dictionary, sort_topics=False)
#     pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
#     st.components.v1.html(pyLDAvis_html, width=width, height=height, scrolling=True)

####################### wordnet #########################

st.title('Wordnet')

class KGConstruction:
    def __init__(self, api_key, filter_list=None, stopwords_file='stopwords-ko.txt'):
        self.api_key = api_key
        self.tagger = self.setup_tagger()
        self.stopwords = self.load_stopwords(stopwords_file, filter_list)

    def setup_tagger(self):
        t = brn.Tagger(self.api_key, "localhost", 5656)
        cust_dic = t.custom_dict("my_dict_01")
        cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
        cust_dic.update()
        t.set_domain("my_dict_01")
        return t

    def load_stopwords(self, stopwords_file, filter_list):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        if filter_list:
            stopwords.extend(filter_list)
        return set(stopwords)

    def preprocess_text(self, lines):
        tokenized_lines = []
        count_dict = {}
        for line in tqdm(lines):
            tagged = self.tagger.tags([line])
            tokens = [word for word, pos in tagged.pos() 
                    if pos in ['NNG', 'NNP'] and word not in self.stopwords and len(word) >= 2]
            tokenized_lines.append(tokens)
            for token in tokens:
                count_dict[token] = count_dict.get(token, 0) + 1
        return tokenized_lines, count_dict

    def train_word2vec_model(self, sentences):
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0)
        return model

    def visualize_word_network(self, model, words, topn=5, filename='wordnet.png'):
        G = nx.DiGraph()
        for word in words:
            similar_words = model.wv.most_similar(word, topn=topn)
            G.add_node(word)
            for similar_word, _ in similar_words:
                G.add_edge(word, similar_word)

        # 시각화 설정
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.5)

        # 노드 색상 및 크기 조정
        node_color = [G.degree(v) for v in G]
        node_size = [1000 * G.degree(v) for v in G]

        nx.draw(G, pos, with_labels=True, font_family='Malgun Gothic', font_weight='bold', 
                node_color=node_color, node_size=node_size, cmap=plt.cm.Reds, 
                edge_color='lightgray', width=2, arrows=True)
        plt.savefig('wordnet.png')
        plt.close()

def execute_kg_construction(contents, api_key, filter_list, topn=20):
    kg = KGConstruction(api_key, filter_list)
    processed_texts, count_dict = kg.preprocess_text(contents)
    model = kg.train_word2vec_model(processed_texts)
    sorted_words = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_words[:topn]]
    kg.visualize_word_network(model, top_words, filename='wordnet.png')
    st.image('wordnet.png')

if __name__ == "__main__":
    contents = filtered_df['content'].tolist()
    api_key = 'koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA'
    filter_list = ['게임', '던파']
    if st.toggle('Wordnet'):
        execute_kg_construction(contents, api_key, filter_list, topn=10)

### plotly를 활용한 interactive graph ###

# def visualize_word_network_with_plotly(self, model, words, topn=5, filename='wordnet.html'):
#     G = nx.DiGraph()
#     for word in words:
#         similar_words = model.wv.most_similar(word, topn=topn)
#         G.add_node(word)
#         for similar_word, _ in similar_words:
#             G.add_edge(word, similar_word)

#     # 시각화 설정
#     plt.figure(figsize=(12, 10))
#     pos = nx.spring_layout(G, k=0.5)

#     # 노드 색상 및 크기 조정
#     node_color = [G.degree(v) for v in G]
#     node_size = [1000 * G.degree(v) for v in G]

#     nx.draw(G, pos, with_labels=True, font_family='Malgun Gothic', font_weight='bold',
#             node_color=node_color, node_size=node_size, cmap=plt.cm.Reds,
#             edge_color='lightgray', width=2, arrows=True)
#     plt.savefig(filename)
#     plt.close()

########################### n-gram ##################################
    
# n-gram 추출 함수들
def clean_text(text):
    """특수 문자 제거 및 텍스트 정제"""
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def generate_ngrams(tokens, n):
    """주어진 토큰 리스트에서 n-gram 생성"""
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def get_most_common_ngrams(texts, n=2, num=10):
    """텍스트 리스트에서 가장 빈번한 n-gram 반환"""
    all_ngrams = []
    for text in texts:
        cleaned_text = clean_text(text)
        tokens = cleaned_text.split()
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    return Counter(all_ngrams).most_common(num)

# Streamlit 앱
st.title("n-gram 시각화")
texts = filtered_df['content'].tolist()

n = st.slider("n-gram의 n값 선택", 2, 5, 2)
num_ngrams = st.slider("보여줄 n-gram 수 선택", 5, 20, 10)

most_common_ngrams = get_most_common_ngrams(texts, n=n, num=num_ngrams)

# Altair 차트 생성
ngrams, counts = zip(*most_common_ngrams)
source = alt.Data(values=[{'ngram': ngram, 'count': count} for ngram, count in zip(ngrams, counts)])
chart = alt.Chart(source).mark_bar().encode(
    x=alt.X('count:Q', title='빈도'),
    y=alt.Y('ngram:N', title='n-gram', sort='-x')
).properties(
    width=600,
    height=400,
    title=f'가장 빈번한 {n}-gram'
)

if st.toggle('n-gram'):
    st.altair_chart(chart, use_container_width=True)

####################### 검색 #######################

def visualize_data(df):
    # 일자별 댓글 수 시각화
    comments_per_day = df.groupby(df['date'].dt.date)['content'].count().reset_index(name='comments_count')
    comments_chart = alt.Chart(comments_per_day).mark_bar().encode(
        x=alt.X('date:T', axis=alt.Axis(format='%m/%d')),
        y='comments_count:Q',
        tooltip=['date', 'comments_count']
    ).properties(title='일별 댓글 수')
    st.altair_chart(comments_chart, use_container_width=True)

    # 일자별 평균 극성 점수 시각화
    sentiment_per_day = df.groupby(df['date'].dt.date)['weighted_sentiment_score'].mean().reset_index(name='average_sentiment')
    sentiment_chart = alt.Chart(sentiment_per_day).mark_line(point=True).encode(
        x=alt.X('date:T', axis=alt.Axis(format='%m/%d')),
        y='average_sentiment:Q',
        tooltip=['date', 'average_sentiment']
    ).properties(title='일별 감성 점수 평균')
    st.altair_chart(sentiment_chart, use_container_width=True)

# 메인 함수
def main():
    st.title("특정 단어 검색")

    # 검색어 입력 받기
    search_query = st.text_input("내용을 입력하세요:")

    # 검색어가 입력된 경우에만 실행
    if search_query:
        # 검색어를 포함하는 댓글 필터링
        search_results = filtered_df[filtered_df['content'].str.contains(search_query, case=False, na=False)]

        # 결과 데이터프레임이 비어있지 않은 경우 시각화
        if not search_results.empty:
            st.write(f"'{search_query}' : {len(search_results)}개")
            visualize_data(search_results)
        else:
            st.write(f"검색 결과 없음")

if __name__ == "__main__":
    main()