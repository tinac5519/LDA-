# LDA-
可以製作詞雲、分群、以及使用coherence、perplexity的方式計算在哪個主題數量時的精確度最高
本程式分為三個階段，分別為爬取網路文章之爬蟲程式，每兩日從資料庫下載並進行斷詞之斷詞程式(雖然亦可將其與爬蟲程式寫在一起，但分開可降低系統耗能)，以及最後進行LDA的程式。
本程式使用之資料庫為Google Big Query，執行則可寫成function或是使用colab進行。

# 爬蟲程式
此處以PTT美妝版為例

```
#import resource
import json
import requests
import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from bs4 import BeautifulSoup
from datetime import datetime
import re
```

若使用Colab則需先進行認證
```
from google.colab import auth
auth.authenticate_user()
```

因為只爬取當日文章，所以先爬一次日期，再爬取文章，並設立條件句防止爬取空白文。

```
project_id='your_project_id'
table_id = 'your_table_id'
PTT_URL = ["https://www.ptt.cc/bbs/MakeUp/index.html"]
for url in PTT_URL:
    resp = requests.get(url, headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"})
    soup = BeautifulSoup(resp.text, 'html.parser')
    df = pd.DataFrame(columns=['date','media_source','cat_source','art_link','title','content','ws'])
    content = []
    for item in soup.select('div.bbs-screen > div.r-ent'):
      pubtime = item.select('div.meta > div.date')[0].text
      title = re.sub(r'[\t\r\n]','',item.select('div.title')[0].text)
      if str(datetime.today().date().strftime('%m/%d')) == pubtime:
        link_suffix = item.select('div.title > a')[0]['href']
        art_link = 'https://www.ptt.cc'+link_suffix
        resp1 = requests.get(art_link, headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"})
        soup1 = BeautifulSoup(resp1.text, 'html.parser')
        para = soup1.select('div.bbs-screen')[0].findAll(text=True,recursive=False)[0]
        if re.sub(r'[\t\r\n]','',para) != '':
          new_row = {'date': str(datetime.today().year)+'/'+pubtime, 'media_source': 'PTT/Makeup', 'cat_source': url, 'art_link': art_link, 'title':title, 'content': re.sub(r'[\t\r\n]','',para), 'ws': None }
          df=df.append(new_row,ignore_index=True)
        else:
          break
df['timestamp'] = pd.to_datetime(df['date'],format='%Y/%m/%d')
pandas_gbq.to_gbq(df,table_id,project_id=project_id,progress_bar=True,if_exists='append')
```
#斷詞
此處使用結巴斷詞系統，雖然個人覺得斷詞與爬蟲在一起更為恰當，但因系統運算考量故分開進行。
此處頻率為每兩日將文章斷詞一次。

一樣先載入資源
```
import jieba.posseg
import jieba
import json
import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from datetime import datetime
```

把不要的詞性跟詞彙剃除
```
stopwords = ['，','。','XD','XDD']
stop = ['v','vx','p','pba','pbei','c','cc','e','h','j','k','m','mg','mq','q','r','rr','u','x','y','ud','ug','uj','ul','uv','uz']
```

把資料從資料庫download下來，斷詞完後再塞回資料庫。

```
project_id='your_project_id'
query="""
  SELECT content
  FROM `your_project_id.ypur_table`
  WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
  """
output = pd.read_gbq(query=query, project_id=project_id)
products_list_temp = output.content.tolist()
products_list = []
for i in products_list_temp:
  seg_lig = jieba.posseg.cut(i)
  con=[]
  for word, flag in seg_lig:
      if (flag not in stop) and (word not in stopwords) and (word!=''):
        con.append(word)
    products_list.append(con)
output['ws'] = products_list
data_json=output.to_dict('records')
client = bigquery.Client(project='dvibe-main')
# Iterate through the fetched rows and update ws values
for row in data_json:
    content = row['content']
    
# Perform necessary operations to obtain ws value using Python code
ws = str(row['ws'])
    
# Update the ws column in the BigQuery table
update_query = """
  UPDATE `your_project_id.ypur_table`
  SET ws = @ws
  WHERE content = @content
    """
parameters = [
        bigquery.ScalarQueryParameter("ws", "STRING", ws),
        bigquery.ScalarQueryParameter("content", "STRING", content)
    ]
update_job_config = bigquery.QueryJobConfig()
update_job_config.query_parameters = parameters 
update_job = client.query(update_query, job_config=update_job_config)
update_job.result()
```

#LDA
進行LDA主題分析，可將分析主題輸出成html格式，並儲存於cloud_storage上。
此處為每月執行一次。

import resource
```
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import json
from google.cloud import bigquery
import pandas_gbq
import requests
import json
import ast
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn
from google.cloud import storage
from datetime import date, timedelta
```

再設一次停用詞以防有漏網之魚
```
stopwords = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]
```

因執行地點為cloud function，效能緣故，故將判斷混淆度之主題數量設為4-8(for i in range(4, 9))，若情況允許可設定更多主題數量，提升準確度。

```
project_id='dvibe-main'
query="SELECT ws FROM `dvibe-main.word_analyzer.ptt_makeup_content` WHERE DATE_TRUNC(DATE(timestamp), MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)"
output = pd.read_gbq(query=query,project_id=project_id)
products_list_temp = output.ws.tolist()
products_list=[]
for i in products_list_temp:
    try:
      a = ast.literal_eval(i)
    except:
      a=''
    products_list.append(a)
    text=[]
for i in products_list:
    temp=''
    for k in i:
      if (k not in stopwords)and len(k)>1:
        temp+=k
        temp+=' '
    text.append(temp)
output["content_cutted"] = text
n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                #stop_words=,
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(output.content_cutted)
perplexity_score = []
tpoic_number = []
for i in range(4, 9):
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tf)
    lda.perplexity(tf, sub_sampling=False)
    perplexity_score.append(lda.perplexity(tf, sub_sampling=False))
    tpoic_number.append(i)
a=perplexity_score.index(min(perplexity_score))
num_topics = tpoic_number[a]

n_topics = num_topics
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
p = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
a = pyLDAvis.prepared_data_to_html(p, d3_url=None, ldavis_url=None, ldavis_css_url=None, template_type="general", visid=None, use_http=False)
today = date.today().replace(day=1) - timedelta(days=1)
month = today.strftime("%Y-%m")
storage_client = storage.Client()
content_type = 'text/html'
bucket = storage_client.bucket('lda_html')
blob = bucket.blob(month)
blob.upload_from_string(a, content_type='text/html')
``
