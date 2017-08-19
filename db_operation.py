import sys

from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from tqdm import tqdm

def insert_collection(db, collection):
    '''
    mongodb的 save() 和 insert() 函数都可以向 collection 里插入数据，但两者是有两个区别：

    1.save() 函数实际就是根据参数条件, 调用了 insert() 或 update() 函数.
    如果想插入的数据对象存在, insert() 函数会报错, 而 save() 函数则是相当于使用 update() 函数，改变原来的对象;
    如果想插入的对象不存在, 那么它们执行相同的 insert() 函数插入操作.
    这里可以用几个字来概括它们两的区别, 即所谓："有则改之,无则加之".

    2.insert() 可以一次性插入一个列表，而不用遍历，效率高; save() 则需要遍历列表，一个个插入。
    
    db.collection.insert_one({'attribute': value})
    
    '''

    return

def update_collection(db, collection):
    '''
    update(criteria, objNew, upsert, mult)
    - criteria: 查询文档，用于定位需要更新的目标文档。
    - objNew: 修改器文档，用于说明要对找到的文档进行哪些修改。
    - upsert: 如目标记录不存在，是否插入新文档。
    - multi: 是否更新多个文档。
    
    ---------
    # '$set' & '$addToSet'
    db.collection.update({'attribute1': value1, 'attribute2': value2}, \
                        {'$set': {'attribute1': new_value1}, '$set':{'attribute2': new_value2}, \
                        '$addToSet':{'attribute3': value3}}, upsert=True)
    # 其中 '$set' 表示对原来的记录进行修改，'$addToSet' 表示添加 'attribute3' 字段到 
    # {'attribute1'=new_value1, 'attribute2'=new_value2} 这条记录中。
    # '$addToSet'和'$push'类似，不过仅在该元素不存在时才添加 (Set 表示不重复元素集合)。
    --------- 
    # '$unset': 移除字段属性。
    db.collection.update({'attribute1': value1}, {'$unset': {'attribute2': value2}, '$unset': {'attribute3': value3}})
    ---------
    # '$push' & 'pushAll'
    db.collection.update({'attribute1': value1}, {'$push': {'attribute2': value2}})
    db.collection.update({'attribute1': value1}, {'$pushAll': {'attribute2': value2}})
    ---------
    # '$each': 添加多个元素。
    db.collection.update({'attribute1': value1}, {'$addToSet': {'attribute2': {'$each': [1,2,3,4]}}})
    ---------
    # '$pop': 按照 index 位置下标移除元素。
    # 原先记录显示（忽略 '_id'）：{'attribute1': [1, 2, 3, 4, 5, 6, 7, 2, 3], 'attribute2': value}

    db.collection.update({'attribute2': value}, {'$pop': {'attribute1': 1}}) # 移除最后一个元素
    # 此刻字段显示：{'attribute1': [1, 2, 3, 4, 5, 6, 7, 2], 'attribute2': value}

    db.collection.update({'attribute2': value}, {'$pop':{'attribute1': -1}}) # 移除第一个元素
    # 此刻字段显示：{'attribute1': [2, 3, 4, 5, 6, 7, 2], 'attribute2': value}
    ---------
    # '$pull': 按值移除元素。
    # '$pullAll': 移除所有符合条件的元素。
    
    # 原先字段显示（忽略 '_id'）： {'attribute1': [2, 3, 4, 5, 6, 7, 2], 'attribute2': value}
    db.collection..update({'attribute2': value}, {'$pull':{'attribute1': 2}}) # 移除全部 2
    # 此刻字段显示：{'attribute1': [3, 4, 5, 6, 7], 'attribute2': value}

    db.collection.update({'attribute2': value}, {'$pullAll':{'attribute1': [3,5,6]}}) # 移除 3,5,6
    # 此刻字段显示：{'attribute1': [4, 7], 'attribute2': value}
    ---------
    '''

    return

def remove_collection(db, collection):
    '''
    db.collection.remove() # 表示删除集合里的所有记录
    db.collection.remove({'attribute': value}) # 表删除某属性 attribute=value 的记录

    id = db.collection.find_one({'attribute': value})['_id']
    db.collection.remove(id) # 查找到某属性 attribute=value 的记录，并根据记录的 id 删除该记录
    db.collection.drop() # 表示删除整个集合
    * 删除文档通常很快，但是如果要清空整个集合，那么使用 drop() 直接删除集合会更快（然后在这个空集合上重建各项索引）。
    '''

    return


def query_collection(db, collection):
    '''
    数据库的查询基本是通过 find() 函数进行查询，其中大于、大于等于、小于、小于等于这些关系运算符经常要用到，分别用'$gt','$gte','$lt','$lte'表示。
    
    db.collection.find({'attribute': {"$lt":15}}) # 查找符合某属性 attribute 的值小于 15 的多条记录
    db.collection.find({'attribute': value}) # 查找符合某属性 attribute=value 的多条记录，查不到时返回 None
    db.collection.find_one({'attribute': value}) # 只查找某属性 attribute=value 的一条记录，查不到时返回 None
    
    ---------
    # 只显示集合中的所有记录的 attribute1、attribute2 属性值, '_id': 0 表示一般忽略不显示 _id 的值， 'attribute': 1 表示显示该字段
    # 如果不指定，是默认显示所有字段（包括 _id ）
    for item in db.colleciton.find({}, {'_id': 0, 'attribute1': 1, 'attribute2': 1}): print item

    # 显示集合中所有 attribute=21 的记录的 attribute1、attribute2 属性值
    for item in db.collection.find({'attribute2': 21}, {'_id': 0, 'attribute1': 1, 'attribute2': 1}): print item
    ---------
    # 查找符合属性 12<attribute1<15, attribute2=value 的多条记录，查不到时返回 None
    for item in db.colleciton.find({'attribute1': {'$gt': 12, '$lt': 15}, 'attribute2': value}): print item
    
    # 查找符合属性 attribute1=21, attribute2=value 的多条记录，查不到时返回 None
    for item in db.colleciton.find({'attribute1': 21, 'attribute2': value}): print item
    ---------
    # Exists
    # 查找存在属性 attribute 的所有记录
    db.collection.find({'attribute': {'$exists':True}})
    
    # 查找不存在属性 attribute 的所有记录 
    db.collection.find({'attribute': {'$exists':False}})
    ---------
    # IN
    # 查找符合属性 attribute 等于 (23, 26, 32) 的多条记录，查不到时返回 None
    for item in db.collection.find({'attribute': {'$in': (23, 26, 32)}}): print item 
    # 查找符合属性 attribute 不等于 (23, 26, 32) 的多条记录，查不到时返回 None
    for item in db.collection.find({'attribute': {'$nin': (23, 26, 32)}}): print item 
    
    # IN 与 查询制定字段结合
    for item in db.collection.find({'attribute': {'$in': (value1, value2)}}, {'_id': 0, 'attribute': 1})
    --------- 
    # OR
    for item in db.collection.find({"$or":[{"age":25}, {"age":28}]}): print item
    for item in db.collection.find({"$or":[{"age":{"$lte":23}}, {"age":{"$gte":33}}]}): print item
    
    # OR 与 查询制定字段结合
    for item in db.collection.find({'$or': [{'attribute': value1}, {'attribute': value2}]}, {'_id': 0, 'attribute': 1})
    ---------
    # 判断数组属性是否包含全部条件
    for item in db.collection.find({'attribute': {'$all': (23, 26, 32)}}): print item
    * 注意和 $in 的区别。$in 是检查目标属性值是条件表达式中的一员，而 $all 则要求属性值包含全部条件元素。
    ---------
    # 正则表达式查询
    # 查询出 attribute 为 'value1', 'value3', 'value5' 的记录
    for item in db.collection.find({'attribute': {'$regex' : r'(?i)value[135]'}}, {'_id': 0, 'attribute': 1}): 
        print item 
    ---------
    # 匹配数组属性元素数量
    for item in db.collection.find({'attribute': {'$size': 3}}, {'_id': 0, 'attribute': 1}): print item
    ---------
    # 数据类型转换
    对于一条记录 x，若其字段 'attribute' 为 <string> 型，则可以如下转换为 <int> 型。
    x['price']=int(x['price'])

    $type: 用于判断属性类型。
        for item in db.collection.find({'attribute': {'$type':1}}): print item # 查询数字类型的
        for item in db.collection.find({'attribute': {'$type':2}}): print item # 查询字符串类型的
    各种类型值的代表值:
    double:1    string: 2   object: 3   array: 4    binary data: 5
    object id: 7    boolean: 8  date: 9 null: 10
    ---------
    # 计数
    print(db.collection.find().count()) 
    print(db.collection.find({'attribute': {'$gt':30}}).count()) 
    ---------
    # 排序，对记录进行排序，用 sort() 函数，形如 find().sort([('attribute',1/-1)]) 表示按某属性 attribute 的升序/降序排列
    pymongo.ASCENDING # 表按升序排列，也可以用 1 来代替
    pymongo.DESCENDING #表按降序排列， 也可以用 -1 来代替
    
    for item in db.collection.find().sort([('attribute', pymongo.ASCENDING)]): print item
    for item in db.collection.find().sort([('attribute', pymongo.DESCENDING)]): print item
    
    for item in db.collection.find().sort([('attribute1', pymongo.ASCENDING), ('attribute2', pymongo.DESCENDING)]): 
        print item 
    
    for item in db.collection.find(sort = [('attribute1', pymongo.ASCENDING), ('attribute2', pymongo.DESCENDING)]): 
        print item
    
    # 指定字段条件查找 + 排序 + 指定字段显示
    for item in db.collection.find({'attribute1': value}, {'_id': 0, 'attribute1': 1, 'attribute2': 1}) \
                                .sort([('attribute1', 1), ('attribute2', -1)]):  print item
    * 可能会出现 「OperationFailed: Sort operation used more than the maximum bytes of RAM. 
    * Add an index, or specify a smaller limit.」 的错误，需要在后面添加一个 limit()
    
    for item in db.collection.find({'attribute1': value}, {'_id': 0, 'attribute1': 1, 'attribute2': 1}) \
                                .sort([('attribute1', 1), ('attribute2', -1)]).limit(100):  print item
    * 比如，我们要做一个排行榜功能，需要在某 collection 中查找分数最多的 100 名玩家。
    ---------
    # 从第几行开始读取(SLICE)，读取多少行(LIMIT)
    # 从第2行开始读取，读取3行记录
    for item in db.collection.find().skip(2).limit(3): print item
    for item in db.collection.find(skip=2, limit=3): print item
    for item in db.collection.find({}, {'_id': 0, 'attribute1': 1}, skip=2, limit=3): print item
    ---------
    '''

    return

# 根据信息文件建立表内容.
def extract_attribute(file_list):
    for i in range(len(file_list)):
        if i == 0:  # file_list[0] 为 「id.txt」.
            with open(file_list[i], 'r') as locals()['file_' + str(i + 1)]:
                locals()['attribute_' + str(i + 1)] = []
                for eachline in locals()['file_' + str(i + 1)]:
                    line = eachline.strip()
                    locals()['attribute_' + str(i + 1)].append(line)
        else:
            with open(file_list[i], 'r') as locals()['file_' + str(i + 1)]:
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
    for i in range(len(file_list)):
        result.append(locals()['attribute_' + str(i + 1)])

    return result

def create_collection(collection, file_list):
    result = extract_attribute(file_list)

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


file_list = ['features[0].txt', 'features[5].txt', 'features[6].txt',
             'features[7].txt', 'features[8].txt', 'features[9].txt']

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

if __name__ == "__main__":
    main()

# create_collection(collection=collection, file_list=file_list)
