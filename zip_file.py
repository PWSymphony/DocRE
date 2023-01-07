"""
-*- coding: utf-8 -*-
Time    : 2023/1/7 13:12
Author  : Wang
"""
import os
import sys
import tarfile

NOT_ZIP = ['.git', '.pt', '.ckpt']


def file_filter(tarinfo):
    flag = any(tarinfo.name.endswith(token) for token in NOT_ZIP)
    if flag:
        return None
    else:
        return tarinfo


def main():
    cur_dir = os.path.split(sys.path[0])[-1]
    with tarfile.open(f'{cur_dir}.tar', 'w') as tar:
        tar.add(name='.', recursive=True, filter=file_filter)


if __name__ == "__main__":
    main()
