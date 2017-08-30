# Randolph's Package
#
# Copyright (C) 2015-2017 Randolph
# Author: Randolph <chinawolfman@hotmail.com>

import re
import sys
import os
import os.path
import linecache
import chardet
import codecs
import pip

from subprocess import call
from math import log
from collections import defaultdict, Counter
from functools import reduce
from itertools import islice
from six import text_type


def upgrade_package():
    """Upgrade all installed python3 packages."""
    for dist in pip.get_installed_distributions():
        call("pip3 install --upgrade " + dist.project_name, shell=True)


def create_list(num, prefix='', postfix='', filetype='.txt'):
    """
    Create the file list.
    :param num: The number of the file
    :param prefix: The prefix of the file
    :param postfix: The postfix of the file
    :param filetype: The file type of the file
    """
    output_file_list = []
    for i in range(num):
        output_file_list.append(prefix + str(i + 1) + postfix + filetype)
    return output_file_list


def list_cur_all_file():
    """Return a list containing the names of the files in the directory(including the hidden files)."""
    file_list = [filename for filename in os.listdir(os.getcwd())]
    print(file_list)
    return file_list


def listdir_nohidden():
    """Return a list containing the names of the files in the directory."""
    file_list = [filename for filename in os.listdir(os.getcwd()) if not filename.startswith('.')]
    print(file_list)
    return file_list


def extract(input_file, output_file):
    """Extract the first column content of the file to the new file."""
    lines = [eachline.strip().split('\t')[0] for eachline in open(input_file, 'r')]
    open(output_file, 'w').write(''.join((item + '\n') for item in lines))


def judge(input_file1, input_file2):
    """To determine whether the first column content of the two files exist duplicate information."""
    lines_1 = [eachline.strip().split('\t')[0] for eachline in open(input_file1, 'r')]
    lines_2 = [eachline.strip().split('\t')[0] for eachline in open(input_file2, 'r')]
    count = 0
    for item in lines_1:
        if item in lines_2:
            count += 1
        else:
            print(item)
    if count > 0:
        print('Total same info number: %d' % count)
    else:
        print('Exactly the same content.')


def copy_files(source_dir, target_dir):
    """Copy all files of the source path to the target path."""
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)
        if os.path.isfile(source_file):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not os.path.exists(target_file) or (os.path.exists(target_file) and (
                    os.path.getsize(target_file) != os.path.getsize(source_file))):
                open(target_file, "wb").write(open(source_file, "rb").read())
            if os.path.isdir(source_file):
                copy_files(source_file, target_file)


def remove_file_in_first_dir(target_dir):
    """Delete all files of the target path."""
    for file in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file)
        if os.path.isfile(target_file):
            os.remove(target_file)


class Text(object):
    """
    This module brings together a variety of functions for
    text analysis, and provides simple, interactive interfaces.
    """
    def __init__(self, filename):
        self.filename = filename

    @property
    def cur_path(self):
        """Returns the path of the file."""
        return os.path.abspath(self.filename)

    @property
    def line_num(self):
        """Returns the line number of the file."""
        count = 1
        with open(self.filename, 'rb') as fin:
            while True:
                buffer = fin.read(8192 * 1024)
                if not buffer:
                    break
                count += buffer.decode('utf-8').count('\n')
        return count

    def tokens(self, keep_punctuation=None):
        """Return the word tokens of the file content."""
        if keep_punctuation is None:
            regex = '\W+'
        else:
            regex = '(\W+)'
        with open(self.filename, 'r') as fin:
            tokens = []
            for eachline in fin:
                line = re.split(regex, eachline)
                for item in line:
                    if item != ' ' and item != '.\n':
                        tokens.append(item)
        return tokens

    def sentence_tokens(self, keep_punctuation=None):
        """Return the sentence tokens of the file content."""
        if keep_punctuation is None:
            regex = '\W+'
        else:
            regex = '(\W+)'
        with open(self.filename, 'r') as fin:
            sentence_tokens = []
            for eachline in fin:
                line = []
                eachline = re.split(regex, eachline)
                for item in eachline:
                    if item != ' ' and item != '.\n':
                        line.append(item)
                sentence_tokens.append(line)
        return sentence_tokens

    def get_after_lines_content(self, n):
        """Get the content after the N line of the file."""
        string = linecache.getlines(self.filename, n)
        return string

    def get_line_content(self, n):
        """Get the N line content of the file."""
        string = linecache.getline(self.filename, n)
        return string

    # 根据某列(index)内容排序, index 表示需要根据哪一列内容进行排序
    def sort(self, index):
        """
        Sort the file content according to the 'index' row to a new file.
        :param index: The row of the file content need to sort
        """
        sorted_lines = sorted(open(self.filename, 'r'), key=lambda x: float(x.strip().split('\t')[index]), reverse=True)
        open(self.filename, 'w').write(''.join(sorted_lines))

    def detect_file_encoding_format(self):
        """Detect the encoding of the given file."""
        with open(self.filename, 'rb') as f:
            data = f.read()
        source_encoding = chardet.detect(data)
        print(source_encoding)
        return source_encoding

    def convert_file_to_utf8(self):
        """Convert the encoding of the file to 'utf-8'(does not backup the origin file)."""
        with open(self.filename, "rb") as f:
            data = f.read()
        source_encoding = chardet.detect(data)['encoding']
        if source_encoding is None:
            print("??", self.filename)
            return
        print("  ", source_encoding, self.filename)
        if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
            content = data.decode(source_encoding, 'ignore')  # .encode(source_encoding)
            codecs.open(self.filename, 'w', encoding='utf-8').write(content)

    def concordance(self, word, width=79, lines=25):
        """
        Print a concordance for ``word`` with the specified context window.
        Word matching is not case-sensitive.
        """
        half_width = (width - len(word) - 2) // 2
        context = width // 4  # approx number of words of context

        tokens = self.tokens(True)
        offsets = []
        for index, item in enumerate(tokens):
            if item == word:
                offsets.append(index)

        if offsets:
            lines = min(lines, len(offsets))
            print("Displaying %s of %s matches:" % (lines, len(offsets)))
            for i in offsets:
                if lines <= 0:
                    break
                left = (' ' * half_width +
                        ' '.join(tokens[i - context:i]))
                right = ' '.join(tokens[i + 1:i + context])
                left = left[-half_width:]
                right = right[:half_width]
                print(left, tokens[i], right)
                lines -= 1
        else:
            print("No matches")
