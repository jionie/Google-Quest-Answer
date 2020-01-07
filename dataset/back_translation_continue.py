import http.client
import hashlib
import json
import urllib
import random
import time
from tqdm import tqdm
import pandas as pd
 
def baidu_translate(content, fromLang='en', toLang='zh'):
    appid = '' # 需要替换
    secretKey = '' # 需要替换
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = content
    #fromLang = 'zh'  # 源语言
    #toLang = 'en'  # 翻译后的语言
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
 
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
 
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")  
        js = json.loads(jsonResponse)  
        #print(js)
        dst = str(js["trans_result"][0]["dst"])  
        #print(dst)  
        return dst
    except Exception as e:
        print('err:' + e)
    finally:
        if httpClient:
            httpClient.close()

def back_translation(content, rawLang='en', tmpLang='zh'):
    content_backup = content
    try:
        content = baidu_translate(content, fromLang=rawLang, toLang=tmpLang)
        time.sleep(0.1)
        content = baidu_translate(content, fromLang=tmpLang, toLang=rawLang)
        time.sleep(0.1)
    except:
        print('error')
        pbar.update(1)
        return content_backup
    
    pbar.update(1)
    return content    
 
if __name__=='__main__':
    df = pd.read_csv('./68.csv')
    now = int(len(df)*0.68)
    pbar = tqdm(total=(len(df)-now)*3)
    t = df['question_title'].iloc[now:]
    q = df['question_body'].iloc[now:]
    a = df['answer'].iloc[now:]
    df.loc[now:,'t_aug'] = t.apply(back_translation)
    df.loc[now:,'q_aug'] = q.apply(back_translation)
    df.loc[now:,'a_aug'] = a.apply(back_translation)
    pbar.close()
    df.to_csv('./train_aug.csv', index=False)
    

