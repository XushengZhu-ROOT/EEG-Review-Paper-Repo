#!/bin/bash

# 脚本功能：1. 把每一个subject的文件放到一个文件夹 2. 创建seed_data文件夹并分类到train/val/test

cd "$(dirname "$0")"
SOURCE_DIR="neurogpt"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误：源目录 $SOURCE_DIR 不存在"
    exit 1
fi

# 创建seed_data文件夹结构
SEED_DATA_DIR="seed_data"
mkdir -p "$SEED_DATA_DIR/train"
mkdir -p "$SEED_DATA_DIR/val"
mkdir -p "$SEED_DATA_DIR/test"

echo "已创建 seed_data 文件夹结构"
echo ""

# 获取所有唯一的subject名称（格式：subject_XX）
all_subjects=$(ls "$SOURCE_DIR" | sed 's/_video.*//' | sort -u)

echo "找到以下subjects："
echo "$all_subjects"
echo ""

# 为每个subject创建文件夹并移动文件
for subject in $all_subjects; do
    echo "处理 $subject ..."
    
    # 创建临时subject文件夹
    mkdir -p "$subject"
    
    # 将所有属于该subject的文件移动到对应文件夹
    find "$SOURCE_DIR" -name "${subject}_*.pickle" -exec mv {} "$subject/" \;
    
    echo "  $subject 文件已移动完成"
    
    # 根据subject编号分类到train/val/test
    subject_num=$(echo "$subject" | sed 's/subject_//' | sed 's/^0*//')
    
    if [ -z "$subject_num" ]; then
        subject_num=0
    fi
    
    if [ "$subject_num" -ge 1 ] && [ "$subject_num" -le 18 ]; then
        # subject 1-18 放到 train
        echo "  移动 $subject 到 train 文件夹"
        mv "$subject" "$SEED_DATA_DIR/train/"
    elif [ "$subject_num" -eq 19 ]; then
        # subject 19 放到 val
        echo "  移动 $subject 到 val 文件夹"
        mv "$subject" "$SEED_DATA_DIR/val/"
    elif [ "$subject_num" -eq 20 ]; then
        # subject 20 放到 test
        echo "  移动 $subject 到 test 文件夹"
        mv "$subject" "$SEED_DATA_DIR/test/"
    else
        echo "  警告：$subject 不在分类范围内（1-20），保留在原位置"
    fi
done

echo ""
echo "全部完成！"
echo "seed_data 文件夹结构："
echo "  train/ - subject_1 到 subject_18"
echo "  val/   - subject_19"
echo "  test/  - subject_20"

