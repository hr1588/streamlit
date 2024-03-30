import pandas as pd
import streamlit as st
import altair as alt
import json
import matplotlib.pyplot as plt
import seaborn as sns
import bareunpy as brn
from collections import Counter
import re
from tqdm import tqdm
from gensim import corpora
import gensim
import pyLDAvis.gensim_models
import networkx as nx
from gensim.models import Word2Vec

df = pd.read_csv('weight_df.csv')
df['date'] = pd.to_datetime(df['date'])

# 스트림릿 앱 설정
st.title('Sentiment Analysis')
st.caption('''사용 데이터  \n\n
           2024년 2월 28일 ~ 2024년 3월 28일 유튜브 헌터, 비질란테 연관 동영상 댓글
    3월 13일 던파 개발자 노트 댓글(https://df.nexon.com/community/news/devnote/2841792)
           ''')
st.caption('왼쪽 sidebar에서 감성과 날짜를 선택하세요.')

st.subheader('Dataframe')
st.caption('sentiment별 좋아요 Top5 댓글')
st.caption('confidence : 감성 분류 확률 / weighted_sentiment_score : 감성 점수 * log(좋아요 수)')

# 사이드바 설정: 감성 선택
sentiment_option = st.sidebar.selectbox(
    'Choose sentiment',
    ('all', 'positive', 'neutral', 'negative')
)

# 사이드바 설정: 날짜 선택 (pd.Timestamp로 변환)
start_date = pd.Timestamp(st.sidebar.date_input('Start date', value=pd.to_datetime('2024-02-28')))
end_date = pd.Timestamp(st.sidebar.date_input('End date', value=pd.to_datetime('2024-03-28')))

# 선택된 감성과 날짜에 따라 데이터 필터링
if sentiment_option == 'all':
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
else:
    filtered_df = df[(df['sentiment'] == sentiment_option) & 
                     (df['date'] >= start_date) & 
                     (df['date'] <= end_date)]

sorted_df = filtered_df.sort_values(by='likes', ascending=False)
if st.toggle('display?'):
    st.table(sorted_df.head())

# 일자별로 그룹화하고, 각 그룹의 content 개수(댓글 수)를 계산
comments_per_day = filtered_df.groupby('date')['content'].count().reset_index().rename(columns={'content': 'comments_count'})

# 알테어 차트 생성, 날짜 표시 형식을 '월/일'로 변경
chart = alt.Chart(comments_per_day).mark_bar().encode(
    x=alt.X('date:T', title='Date', axis=alt.Axis(format='%m/%d')),  # 날짜 형식을 '월/일'로 지정
    y=alt.Y('comments_count:Q', title='Comments Count'),
    tooltip=[alt.Tooltip('date:T', title='Date', format='%m/%d'), alt.Tooltip('comments_count:Q', title='Comments Count')]  # 툴팁에서도 날짜 형식을 '월/일'로 지정
).properties(
    title='Daily Comments Count',
    width=600,
    height=400
)

# 스트림릿에 차트 출력
st.altair_chart(chart, use_container_width=True)

st.markdown('## 점수 분포 시각화')

# 감정 점수 분포를 위한 알테어 바 차트 생성
hist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X("weighted_sentiment_score:Q", bin=True, title='Sentiment Score'),
    alt.Y('count()', title='Frequency'),
    # 각 막대의 텍스트 레이블을 위한 Tooltip 추가 (선택적)
    tooltip=[alt.Tooltip('count()', title='Frequency')]
).properties(
    title='1. Sentiment Score Distribution'
)

# 막대 위에 빈도수를 표시하는 텍스트 레이블 추가
text = hist.mark_text(
    align='center',
    baseline='bottom',
    dy=-5  # 텍스트 위치를 막대 위로 조정
).encode(
    text='count()'  # 빈도수를 텍스트로 표시
)

# 바 차트와 텍스트 레이블 결합
final_chart = hist + text

st.altair_chart(final_chart, use_container_width=True)

# 시간에 따른 평균 감정 점수 계산
average_sentiment_score_by_date = filtered_df.groupby('date')['weighted_sentiment_score'].mean().reset_index()

# 알테어 라인 차트 생성
line_chart = alt.Chart(average_sentiment_score_by_date).mark_line(point=True).encode(
    x=alt.X('date:T', axis=alt.Axis(format='%m/%d', title='Date')),  # 날짜 형식을 '월/일'로 지정
    y=alt.Y('weighted_sentiment_score:Q', title='Average Sentiment Score'),
    tooltip=[alt.Tooltip('date:T', format='%m/%d'), 'weighted_sentiment_score:Q']  # 툴팁에서도 날짜 형식 지정
).properties(
    title='2. Temporal Changes in Average Sentiment Score',
    width=600  # 차트 너비 설정
)

st.altair_chart(line_chart, use_container_width=True)

# 결과 신뢰도 및 기간내 감성 평균점수

st.markdown('## 결과 신뢰도 및 기간내 감성 평균점수')

if sentiment_option == 'all':
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    confience_mean = filtered_df.groupby('sentiment', as_index=False)['confidence'].mean()
    sentiment_mean = filtered_df.groupby('sentiment', as_index=False)['weighted_sentiment_score'].mean()
    st.dataframe(pd.merge(confience_mean, sentiment_mean))
else:
    filtered_df = df[(df['sentiment'] == sentiment_option) & 
                     (df['date'] >= start_date) & 
                     (df['date'] <= end_date)]
    confience_mean = filtered_df.groupby('sentiment', as_index=False)['confidence'].mean()
    sentiment_mean = filtered_df.groupby('sentiment', as_index=False)['weighted_sentiment_score'].mean()
    st.dataframe(pd.merge(confience_mean, sentiment_mean))

## 감성별 

st.markdown('## 감성별 상위 단어 출력')

def tokenizer(data):
    API_KEY = "koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA"
    t = brn.Tagger(API_KEY, "localhost", 5656)
    cust_dic = t.custom_dict("my_dict_01") 
    cust_dic.copy_np_set({'캐릭터','비질란테','박종민','강정호','윤명진','네오플','헌터'})
    cust_dic.update()

    t.set_domain("my_dict_01")
    tags = t.tags(data)
    nouns = tags.nouns()

    with open('stopwords-ko.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [word.strip() for word in stopwords]
        filter_list = ['던파', '게임']
        stopwords = stopwords + filter_list
    
    tokens = [word for word in nouns if word not in stopwords and len(word) >= 2]
    count_tokens = Counter(tokens)
    df_tokens = pd.DataFrame(count_tokens.items(), columns=['word','count'])
    df_tokens = df_tokens[(df_tokens['count']>=10)].sort_values('count', ascending=False).reset_index(drop=True)
    return df_tokens

token_counts_df = tokenizer(filtered_df['content'].tolist())
                            
if st.checkbox('display?'):
    st.dataframe(token_counts_df.head())

# 'count'가 20 이상인 단어만 필터링
filtered_words = token_counts_df[token_counts_df['count'] >= 20]

# 사이드바 옵션이 'All'일 경우와 그렇지 않을 경우를 구분
if sentiment_option == 'All':
    # 감성별 색상 구분이 포함된 차트
    color_encoding = alt.Color('sentiment:N', legend=alt.Legend(title="Sentiment"))
else:
    # 단일 색상 차트
    color_encoding = alt.value('steelblue')  # 단일 색상 설정, 원하는 색상 코드로 변경 가능

# 바 차트 기본 설정, x축과 y축 교환
chart = alt.Chart(filtered_words).mark_bar().encode(
    y=alt.Y('word:N', title='Word', sort=alt.EncodingSortField(field='count', order='descending')),  # y축에 word, 내림차순 정렬
    x=alt.X('count:Q', title='Frequency'),  # x축에 count
    color=color_encoding,  # 조건부 색상 인코딩
    tooltip=['word', 'count']  # sentiment 참조 제거
).properties(
    title='Word Frequency',
    width=600,
    height=800  # 차트의 높이 조정, 필요에 따라 조절
)

# 각 막대 옆에 빈도수 표시하는 텍스트 레이블 추가
text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=3  # 텍스트 위치 오른쪽으로 조정
).encode(
    text='count:Q'
)

# 바 차트와 텍스트 레이블 결합
final_chart = chart + text

# 스트림릿에 차트 출력
st.altair_chart(final_chart, use_container_width=True)


### LDA

# st.title('LDA Topic Modeling')

# def preprocess_text(lines, tagger, stopwords):
#     tokenized_lines = []
#     for line in tqdm(lines):

#         # 형태소 분석 후 명사 추출
#         tagged = tagger.tags([line])
#         nouns = tagged.nouns()  # 명사만 추출, tagged가 여러 문장을 처리할 수 있다고 가정

#         # 한 글자 단어 제외 및 불용어 처리
#         tokens = [word for word in nouns if len(word) > 1 and word not in stopwords]
#         tokenized_lines.append(tokens)

#     return tokenized_lines

# # 불용어 목록 로드
# with open('stopwords-ko.txt', 'r', encoding='utf-8') as f:
#     stopwords = f.readlines()
# stopwords = [word.strip() for word in stopwords]
# filter_list = ['던파', '게임']
# stopwords += filter_list

# def LDA_display(data, num_topics, stopwords):
#     API_KEY = "koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA"
#     t = brn.Tagger(API_KEY, "localhost", 5656)

#     # 사용자 정의 사전 설정은 여기서 계속 유지
#     cust_dic = t.custom_dict("my_dict_01")
#     cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
#     cust_dic.update()
#     t.set_domain("my_dict_01")
    
#     # preprocess_text 함수 호출 시 불용어 목록 추가
#     tokenized_lines = preprocess_text(data, t, stopwords)
#     dictionary = corpora.Dictionary(tokenized_lines)
#     corpus = [dictionary.doc2bow(text) for text in tokenized_lines]

#     lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
#     topics = lda_model.print_topics(num_words=4)
#     for topic in topics:
#         st.caption(f'{topic[0]}번째 토픽: {topic[1]}')
    
#     # pyLDAvis 시각화를 HTML로 변환
#     vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
#     pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)

#     # HTML 컨텐츠를 가운데 정렬하기 위한 스타일을 추가한 HTML 코드
#     # centered_html = f"""
#     # <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
#     #     <div style="width: 1200px;">{pyLDAvis_html}</div>
#     # </div> """

#     # Streamlit에서 HTML 시각화 표시
#     return st.components.v1.html(pyLDAvis_html, width=800, height=800, scrolling=True)

# LDA_display(filtered_df['content'].tolist(),4, stopwords)

### wordnet

class KGConstruction:
    def __init__(self, filter_list, API_KEY):
        self.filter_list = filter_list
        self.API_KEY = API_KEY
        self.tagger = self.setup_tagger()
        self.stopwords = self.load_stopwords()

    def setup_tagger(self):
        # brn.Tagger 인스턴스 생성 및 사용자 정의 사전 설정
        t = brn.Tagger(self.API_KEY, "localhost", 5656)
        # 사용자 정의 사전 구성
        cust_dic = t.custom_dict("my_dict_01")
        cust_dic.copy_np_set({'캐릭터', '비질란테', '박종민', '강정호', '윤명진', '네오플', '헌터'})
        cust_dic.update()
        t.set_domain("my_dict_01")
        return t
    
    def load_stopwords(self):
        # 불용어 목록 로드
        with open('stopwords-ko.txt', 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
        stopwords = [word.strip() for word in stopwords]
        # filter_list에 있는 단어도 불용어 목록에 추가
        stopwords += self.filter_list
        return stopwords

    def preprocess_text(self, lines, verb=False):
        count_dict = {}
        tokenized_lines = []
        for line in tqdm(lines):
            tagged = self.tagger.tags([line])
            tokens = []
            for p in tagged.pos():
                if p[1] == 'NNG' or p[1] == 'NNP':
                    cleaned_word = re.sub('[^가-힣]', '', p[0])
                    if cleaned_word not in self.stopwords:
                        tokens.append(cleaned_word)
            if verb:
                verbs = tagged.verbs()
                if verbs:
                    tokens = [verb for verb in verbs if verb not in self.stopwords] + tokens
            for token in tokens:
                count_dict[token] = count_dict.get(token, 0) + 1
            tokenized_lines.append(tokens)
        return tokenized_lines, count_dict

    def train_word2vec_model(self, sentences):
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0)
        print(model.wv.vectors.shape)
        return model

    def visualize_word_network(self, model, words, topn=10):
        plt.rc('font', family='Malgun Gothic')
        plt.figure(figsize=(40, 30))
        G = nx.DiGraph()

        for word in words:
            try:
                word_check = model.wv.most_similar(word, topn=topn)
                G.add_node(word)
                for similar_word, similarity in model.wv.most_similar(word, topn=topn):
                    if similar_word in words:
                        G.add_edge(word, similar_word, weight=similarity)
            except:
                continue

        pos = nx.spring_layout(G)
        edges = list(G.edges())
        weights = []
        for u, v in edges:
            weights.append(G[u][v]['weight'])

        nx.draw(G, pos, edgelist=edges, 
                edge_color='black', 
                with_labels=True, font_family='Malgun Gothic',
                font_weight='bold',
                font_size=50)
        plt.savefig('wordnet.png')

    def word_filter(self, sort_dict, topn=50):
        words = []
        cnt = 0

        for d in sort_dict:
            if cnt >= topn:
                break
            else:
                if d[0] not in self.filter_list and len(d[0]) != 1:
                    words.append(d[0])
                    cnt += 1
        return words
    
def print_KG(data, count, filter_list):
    kg = KGConstruction(filter_list=filter_list, API_KEY='koba-5OYXJ5I-CYSEYKQ-WPMR7QY-2BCUYWA')
    processed_texts, cnt_dict = kg.preprocess_text(data) 
    sort_dict = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=True)
    word2vec_model = kg.train_word2vec_model(processed_texts)
    words = kg.word_filter(sort_dict, topn=count)
    return kg.visualize_word_network(word2vec_model, words)

filter_list = ['게임','던파']
print_KG(filtered_df['content'].tolist(), 20, filter_list)
st.image('wordnet.png')