#!/bin/sh
for prompt_type in "P" "B"; do
    random_number=$(date +%s)
    random_number=$(echo $random_number | cut -c 6-10)
    RANDOM=$(date +%s| cut -c 6-10)

    echo "Current prompt_type: $prompt_type"
    echo "生成的随机数是: $random_number"
    echo "生成的随机数是: " $RANDOM
    # 在这里添加你想要执行的其他命令
done