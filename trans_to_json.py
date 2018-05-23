# -*-coding:utf-8 -*-
from pypinyin import pinyin
import json

print(pinyin(u'中心'))
with open('/home/xxy/桌面/a.txt', 'r') as f:
    lines = f.readlines()
json_dic = {}

for line in lines:
    text, pyin = line.strip().split('>')
    print('text:', text, 'pingyin:', pinyin('{}'.format(text)))
    json_dic['{}'.format(text)] = pinyin('{}'.format(text))


json_str = json.dumps(json_dic, ensure_ascii=False)
with open('new.json', 'w') as jf:
    jf.write(json_str)
print(json_dic)

