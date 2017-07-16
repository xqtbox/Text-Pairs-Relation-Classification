# Useful 「file and folder」function collection.
import sys
import os
import os.path
import linecache
import chardet
import codecs
import pip
from subprocess import call

# 创建需要输出的文件列表, prefix 为前缀，默认值为空；filetype 为文件类型，默认值为 .txt 文本类型.
def create_list(num, prefix = '', postfix = '', filetype = '.txt'):
	outputFile_list = []
	for i in range(num):
		outputFile_list.append(prefix + str(i+1) + postfix + filetype)
	return outputFile_list

# 显示当前代码文件所处的绝对路径
def cur_file_dir():
	path = sys.path[0]
	if os.path.isdir(path):	return path
	elif os.path.isfile(path):
		return os.path.dirname(path)

# 显示当前文件下的所有文件（包括隐藏文件）
def list_cur_all_file():
	for filename in os.listdir(cur_file_dir()):
		print(filename)

# 显示当前文件下的可视文件
def listdir_nohidden(path):
	file_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			file_list.append(f)
	print(file_list)
	return file_list

# 计算小文件的行数
def count_line_smallFile(inputFile):
	count = len(open(inputFile, 'r').readlines())
	print(count)
	return count

# 计算大文件（超过1G）的行数，加入enurmerate计数器
def count_line_bigFile(inputFile):
	count = -1
	for count, line in enumerate(open(inputFile, 'r')):
		pass
	count += 1
	print(count)
	return count

# 计算大文件（超过1G）的行数，放入缓存防止内存过载
def count_line_best(inputFile):
	count = 0
	fin = open(inputFile, 'rb')
	while True:
		buffer = fin.read(8192*1024)
		if not buffer:
			break
		count += buffer.decode('utf-8').count('\n')
	fin.close()
	print(count)
	return count

# 提取文件首列信息
def extract(inputFile, outputFile):
	lines = [eachline.strip().split('\t')[0] for eachline in open(inputFile, 'r')]
	open(outputFile, 'w').write(''.join((item + '\n') for item in lines))
	
# 判断两个文件的首列信息是否存在重复信息
def judge(inputFile1, inputFile2):
	lines_1 = [eachline.strip().split('\t')[0] for eachline in open(inputFile1, 'r')]
	lines_2 = [eachline.strip().split('\t')[0] for eachline in open(inputFile2, 'r')]
	count = 0
	for item in lines_1:
		if item in lines_2:	count += 1
		else:
			print(item)
	print(count)

# 获取从第 N 行开始后面所有的信息
def get_line_and_behind(inputFile, N):
	string = linecache.getlines(inputFile, N)
	return string 

# 获取具体某一行的内容
def get_line_content(inputFile, N):
	string = linecache.getline(inputFile, N)
	return string 

# 根据某列(index)内容排序, index 表示需要根据哪一列内容进行排序
def sort(File, index):
	sorted_lines = sorted(open(File, 'r'), key=lambda x: float(x.strip().split('\t')[index]), reverse = True)
	open(File, 'w').write(''.join(sorted_lines))
	
# 某一目录下的所有文件复制到指定目录中	
def copyFiles(sourceDir, targetDir):
	for file in os.listdir(sourceDir): 
		sourceFile = os.path.join(sourceDir, file) 
		targetFile = os.path.join(targetDir, file) 
		if os.path.isfile(sourceFile): 
			if not os.path.exists(targetDir): 
				os.makedirs(targetDir) 
			if not os.path.exists(targetFile) or(os.path.exists(targetFile) \
				and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))): 
				open(targetFile, "wb").write(open(sourceFile, "rb").read()) 
			if os.path.isdir(sourceFile): 
				First_Directory = False
				copyFiles(sourceFile, targetFile)

# 删除一级目录下的所有文件
def removeFileInFirstDir(targetDir): 
	for file in os.listdir(targetDir):
		targetFile = os.path.join(targetDir, file)
		if os.path.isfile(targetFile):
			os.remove(targetFile)

# 检测文件的编码格式
def detect_file_encoding_format(filename):
	with open(filename, 'rb') as f:
		data = f.read()
	source_encoding = chardet.detect(data)
	print(source_encoding)

# 将文件的编码格式转换成'utf-8'
def convert_file_to_utf8(filename):
	# !!! does not backup the origin file
	with open(filename, "rb") as f:
		data = f.read()
	source_encoding = chardet.detect(data)['encoding']
	if source_encoding == None:
		print("??", filename)
		return
	print("  ", source_encoding, filename)
	if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
		content = data.decode(source_encoding, 'ignore') #.encode(source_encoding)
		codecs.open(filename, 'w', encoding='utf-8').write(content)

# 批量升级第三方库
def upgrate_package():
	for dist in pip.get_installed_distributions():
		call("pip3 install --upgrade " + dist.project_name, shell=True)
