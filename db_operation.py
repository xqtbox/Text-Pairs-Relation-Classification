# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys

from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from tqdm import tqdm

files_list = ['features[0].txt', 'features[5].txt', 'features[6].txt',
              'features[7].txt', 'features[8].txt', 'features[9].txt']


def extract_attribute(input_files_list):
    """根据信息文件建立表内容"""
    for i in range(len(input_files_list)):
        if i == 0:  # file_list[0] 为 「id.txt」.
            with open(input_files_list[i], 'r') as locals()['file_' + str(i + 1)]:
                locals()['attribute_' + str(i + 1)] = []
                for eachline in locals()['file_' + str(i + 1)]:
                    line = eachline.strip()
                    locals()['attribute_' + str(i + 1)].append(line)
        else:
            with open(input_files_list[i], 'r') as locals()['file_' + str(i + 1)]:
                locals()['attribute_' + str(i + 1)] = []
                for eachline in locals()['file_' + str(i + 1)]:
                    line = eachline.strip().split('::')
                    new_line = []
                    for index, item in enumerate(line):
                        if index == 0:
                            new_line.append(item)
                        if index != 0 and item != '':
                            new_line.append(item)
                    locals()['attribute_' + str(i + 1)].append(new_line)

    result = []
    for i in range(len(input_files_list)):
        result.append(locals()['attribute_' + str(i + 1)])

    return result


def create_collection(collection, input_files_list):
    result = extract_attribute(input_files_list)

    # 根据项目所使用信息文件的不同，每条数据的 BSON 格式也相应需要修改
    for i in tqdm(range(len(result[0]))):
        data_record = {
            'test_id': result[0][i],
            'test_content': result[1][i],
            'test_option': result[2][i],
            'test_answer': result[3][i],
            'test_analysis': result[4][i],
            'test_knowpoint': result[5][i]
        }
        collection.insert_one(data_record).inserted_id


def main():
    # 建立连接到默认主机（localhost）和端口（27017）。还可以指定主机和/或使用端口：
    try:
        client = MongoClient('localhost', 27017)
        print('Connected Successfully!')
    except ConnectionFailure as e:
        sys.stderr.write('Could not connect to MongoDB: %s' % e)
        sys.exit(1)

    db = client.local
    collection = db['CNN-Sentence-Pairs-Classification-Original']
    collection.update({}, {}, safe=True)
    # create_collection(collection=collection, input_files_list=files_list)


if __name__ == "__main__":
    main()
