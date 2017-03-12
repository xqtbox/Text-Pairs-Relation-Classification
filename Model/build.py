# Tag pairs and build features on them. 
import randolph
import math
from gensim import corpora, models, similarities
from gensim.models import word2vec
from gensim.corpora import TextCorpus, MmCorpus, Dictionary

BASE_DIR = randolph.cur_file_dir()
WORD2VEC_DIR = BASE_DIR + '/math.model'
TEXT_DATA_DIR = BASE_DIR + '/temp.txt' 
DICTIONARY_DIR = BASE_DIR + '/math.dict'

def create_features_list(features_list):
	for i in range(len(features_list)):
		if i == 0:	# features_list[0] 为 「id.txt」.
			with open(features_list[i], 'r') as locals()['feature_' + str(i+1)]:
				locals()['test_' + str(i+1)] = []
				for eachline in locals()['feature_' + str(i+1)]:
					line = eachline.strip()
					locals()['test_' + str(i+1)].append(line)
		else:
			with open(features_list[i], 'r') as locals()['feature_' + str(i+1)]:
				locals()['test_' + str(i+1)] = []
				for eachline in locals()['feature_' + str(i+1)]:
					line = eachline.strip().split('::')
					locals()['test_' + str(i+1)].append(line)
	
	result = []
	for i in range(len(features_list)):
		result.append(locals()['test_' + str(i+1)])
	
	return result
	
def inter(line, features_list, dictionary):
	sentences = word2vec.LineSentence(TEXT_DATA_DIR)
	
	index_front = -1
	index_behind = -1
	list_front_raw = []
	list_behind_raw = []
	
	# features_list[0] 是 id 信息.	
	for index, value in enumerate(features_list[0]):
		if value == line[0]:
			index_front = index
		if value == line[1]:
			index_behind = index
	
#	print(index_front, index_behind)
	
	for i in range(1, len(features_list)):
		locals()['feature' + str(i) + '_front'] = features_list[i][index_front]
		locals()['feature' + str(i) + '_behind'] = features_list[i][index_behind]
	
#	print(locals()['feature' + str(i) + '_front'], locals()['feature' + str(i) + '_behind'])

	def build_front(feature, dictionary, list_raw):
		for item in feature:
			list_raw.append(item)
	
	def build_behind(feature, dictionary, list_raw):
		for item in feature:
			list_raw.append(item)
					
	for i in range(1, len(features_list)):
		build_front(locals()['feature' + str(i) + '_front'], dictionary, list_front_raw)
		build_behind(locals()['feature' + str(i) + '_behind'], dictionary, list_behind_raw)

	list_front = list(list_front_raw)
	list_behind = list(list_behind_raw)
#	list_front.sort()
#	list_behind.sort()

#	print(list_front, list_behind)

	return list_front, list_behind
	
# 根据特征，特征维数构造最终的pair信息。
def build_features(inputFile, outputFile):
	features_inputlist = ['features[0].txt', 'features[5].txt']
	features_list = create_features_list(features_inputlist)
	my_dict = Dictionary.load(DICTIONARY_DIR)
	
	def build(inputlist, string):
		for index, item in enumerate(inputlist):
			if index == (len(inputlist)-1):
				string += item
			else:
				string += item + ' '
		return string
	
	def add_tag(string):
		return string + '<end>' +  ' '
	
	with open(inputFile, 'r') as fin, open(outputFile, 'w') as fout:
		for index, eachline in enumerate(fin):
			print(index)
			line = eachline.strip().split('\t')
			list_front, list_behind = inter(line, features_list, my_dict)
			if line[2] == '0.0':
				outStr = '' + line[0] + '\t' + line[1] + '\t' + '0' + '\t'
			else:
				outStr = '' + line[0] + '\t' + line[1] + '\t' + '1' + '\t'
			outStr = build(list_front, outStr)
			outStr = add_tag(outStr)
			outStr = build(list_behind, outStr)
			fout.write(outStr.strip() + '\n')

	print('All Finished.')
		
build_features('Model1_Training_tag.txt', 'Model1_Training.txt')
#build_features('Model1_Test_few_tag.txt', 'Model1_Test.txt')
#build_features('Model1_Test_total_tag.txt', 'Model1_Test_total.txt')