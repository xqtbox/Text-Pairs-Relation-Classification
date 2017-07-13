from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm

# 建立连接到默认主机（localhost）和端口（27017）。还可以指定主机和/或使用端口：
client = MongoClient('localhost', 27017)
# client = MongoClient('mongodb://localhost:27017')

db = client.local
collection = db['CNN-Sentence-Pairs-Classification-Original']

file_list = ['features[0].txt', 'features[5].txt', 'features[6].txt', 'features[7].txt', 'features[8].txt', 'features[9].txt']

def insert_collection(db, collection):
    '''
    mongodb的 save() 和 insert() 函数都可以向 collection 里插入数据，但两者是有两个区别：

    1.save() 函数实际就是根据参数条件, 调用了 insert() 或 update() 函数.
    如果想插入的数据对象存在, insert() 函数会报错, 而 save() 函数则是相当于使用 update() 函数，改变原来的对象;
    如果想插入的对象不存在, 那么它们执行相同的 insert() 函数插入操作.
    这里可以用几个字来概括它们两的区别, 即所谓："有则改之,无则加之".

    2.insert() 可以一次性插入一个列表，而不用遍历，效率高; save() 则需要遍历列表，一个个插入。
    
    db.collection.insert_one({'attribute: value'})
    
    '''

    return

def update_collection(db, collection):
    '''
    update(criteria, objNew, upsert, mult)
    -criteria: 查询文档，用于定位需要更新的目标文档。
    -objNew: 修改器文档，用于说明要对找到的文档进行哪些修改。
    -upsert: 如目标记录不存在，是否插入新文档。
    -multi: 是否更新多个文档。

    db.collection.update({'gid':last_gid, 'time':l_date}, {'$set':{'gid':last_gid}, '$set':{'time':l_date}, 
                        '$addToSet':{'categories':category_data}}, upsert=True)
    其中 '$set' 表示对原来的记录进行修改，'$addToSet' 表示添加 'categories' 字段到 {'gid'=last_gid, 'time'=l_date} 这条记录中。 
    '''

    return

def remove_collection(db, collection):
    '''
    db.collection.remove() # 表示删除集合里的所有记录
    db.collection.remove({'attribute': 'value'}) # 表删除某属性 attribute=value 的记录

    id = db.collection.find_one({'attribute': 'value'})['_id']
    db.collection.remove(id) # 查找到某属性 attribute=value 的记录，并根据记录的 id 删除该记录
    db.collection.drop() # 表示删除整个集合
    '''

    return


def query_collection(db, collection):
    '''
    数据库的查询基本是通过 find() 函数进行查询，其中大于、大于等于、小于、小于等于这些关系运算符经常要用到，分别用'$gt','$gte','$lt','$lte'表示。
    
    db.collection.find({'attribute': {"$lt":15}}) # 查找符合某属性 attribute 的值小于 15 的多条记录
    db.collection.find({'attribute': 'value'}) # 查找符合某属性 attribute=value 的多条记录，查不到时返回 None
    db.collection.find_one({'attribute': 'value'}) # 只查找某属性 attribute=value 的一条记录，查不到时返回 None
    
    ---------
    # 只显示集合中的所有记录的 attribute1、attribute2 属性值
    for item in db.colleciton.find(fields = ['attribute1', 'attribute2']): print item

    # 显示集合中所有 attribute=21 的记录的 attribute1、attribute2 属性值
    for item in db.collection.find({'attribute2':21}, ['attribute1', 'attribute2']): print item
    * 这里要注意，['attribute1', 'attribute2'] 中可以是一个，也可以是多个；同时 ['attribute1', 'attribute2'] 是放在条件｛｝外的。
    ---------
    # 查找符合属性 12<attribute1<15, attribute2=value, attribute2=value 的多条记录，查不到时返回 None
    for item in db.colleciton.find({'attribute1': {'$gt': 12, '$lt': 15}, 'attribute2': 'value'}): print item
    
    # 查找符合属性 attribute1=21, attribute2=value 的多条记录，查不到时返回 None
    for item in db.colleciton.find({'attribute1': 21, 'attribute2': 'value'}): print item

    * 当使用 find()函数时，对应的条件都是以字典形式表示｛'attribute1': {'$gt':12}, 'attribute2': 'value'｝，有多个条件时，都放在一个｛｝内
    ---------
    # IN
    # 查找符合属性 attribute 等于 (23, 26, 32) 的多条记录，查不到时返回 None
    for item in db.collection.find({'attribute': {'$in':(23, 26, 32)}}): print item 
    # 查找符合属性 attribute 不等于 (23, 26, 32) 的多条记录，查不到时返回 None
    for item in db.collection.find({'attribute': {'$nin':(23, 26, 32)}}): print item 
    --------- 
    # OR
    for item in db.collection.find({"$or":[{"age":25}, {"age":28}]}): print item
    for item in db.collection.find({"$or":[{"age":{"$lte":23}}, {"age":{"$gte":33}}]}): print item
    ---------
    # 计数
    print(db.collection.find().count()) 
    print(db.collection.find({'attribute': {'$gt':30}}).count()) 
    ---------
    # 排序，对记录进行排序，用 sort() 函数，形如 find().sort('attribute',1/-1)，表示按某属性 attribute 的升序/降序排列，注意与后面的 1/-1 是用逗号（，）隔开
    pymongo.ASCENDING # 表按升序排列，也可以用 1 来代替
    pymongo.DESCENDING #表按降序排列， 也可以用 -1 来代替
    
    for item in db.collection.find().sort([('attribute', pymongo.ASCENDING)]): print item
    for item in db.collection.find().sort([('attribute', pymongo.DESCENDING)]): print item
    for item in db.collection.find().sort([('attribute1', pymongo.ASCENDING), ('attribute2', pymongo.DESCENDING)]): print item 
    for item in db.collection.find(sort = [('attribute1', pymongo.ASCENDING), ('attribute2', pymongo.DESCENDING)]): print item
    
    # 组合 + 排序 + 查找
    for item in db.collection.find({'attribute1': 'value'}, sort=[['attribute1',1], ['attribute2',1]], 
                                    fields = ['attribute1', 'attribute2', 'attribute3']): print item
    ---------
    # 从第几行开始读取(SLICE)，读取多少行(LIMIT)
    # 从第2行开始读取，读取3行记录
    for item in db.collection.find().skip(2).limit(3): print item
    for item in db.collection.find(skip=2, limit=3): print item
    ---------
    # 数据类型转换
    对于一条记录 x，若其字段 'attribute' 为 <string> 型，则可以如下转换为 <int> 型。
    x['price']=int(x['price'])

    $type: 用于判断属性类型。
        for item in db.collection.find({'attribute':{'$type':1}}): print item # 查询数字类型的
        for item in db.collection.find({'attribute':{'$type':2}}): print item # 查询字符串类型的
    各种类型值的代表值:
    double:1    string: 2   object: 3   array: 4    binary data: 5
    object id: 7    boolean: 8  date: 9 null: 10
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
                    locals()['attribute_' + str(i + 1)].append(line)

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

# create_collection(collection=collection, file_list=file_list)
