#!/bin/bash

# 1. 配置日志存放目录
LOG_DIR="outputs/auto_run_logs"
mkdir -p "$LOG_DIR"

# 2. 定义需要按顺序执行的命令数组
# 【请注意】：这里以 qwen 模型为例补充了必需的参数，运行前请务必根据实际情况修改
COMMANDS=(
    "python -m train.train_all"
    "python -m train.train_collt_sft --model qwen"
    "python -m train.eval_ablation_clarify --collt_ckpt checkpoints/collt-qwen --base_ckpt Qwen/Qwen2.5-7B-Instruct"
    "python -m train.eval_ablation_tools --ckpt checkpoints/collt-qwen"
    "python -m train.eval_ambiglegalqa --ckpt checkpoints/collt-qwen"
    "python -m train.eval_table3 --ckpt checkpoints/collt-qwen"
    "python -m train.inference_collt --ckpt checkpoints/collt-qwen --query 公司拖欠我三个月工资，我准备辞职，能要求赔偿吗？"
)

# 与上面命令一一对应的日志文件前缀名
LOG_NAMES=(
    "step1_train_all"
    "step2_train_collt_sft"
    "step3_eval_ablation_clarify"
    "step4_eval_ablation_tools"
    "step5_eval_ambiglegalqa"
    "step6_eval_table3"
    "step7_inference_collt"
)

# 3. 定义检测GPU是否空闲的函数
check_gpu_idle() {
    # 获取第一张显卡(索引0)的已用显存 (单位: MB)
    # 如果你有指定的显卡，可以将 -i 0 换成 -i 1 或其他
    local used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    
    # 获取失败时视为忙碌
    if [ -z "$used_mem" ]; then
        return 1 
    fi
    
    # 设定阈值：已用显存小于 1000MB 视为显卡空闲
    if [ "$used_mem" -lt 1000 ]; then
        return 0 # 空闲 (True)
    else
        return 1 # 忙碌 (False)
    fi
}

echo "==================================================="
echo "开始监控 GPU 状态，每 10 分钟检测一次..."
echo "==================================================="

# 4. 轮询监控逻辑
while true; do
    if check_gpu_idle; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): 检测到 GPU 空闲，开始串行执行任务！"
        
        # 获取命令总数
        total_cmds=${#COMMANDS[@]}
        
        for i in "${!COMMANDS[@]}"; do
            cmd="${COMMANDS[$i]}"
            log_file="$LOG_DIR/${LOG_NAMES[$i]}_$(date +%Y%m%d_%H%M%S).log"
            
            echo "---------------------------------------------------"
            echo "$(date '+%Y-%m-%d %H:%M:%S'): 正在执行 ($((i+1))/$total_cmds): $cmd"
            echo "日志将输出至: $log_file"
            
            # 执行命令，并将标准输出和错误输出都重定向到日志文件
            $cmd > "$log_file" 2>&1
            
            # 检查上一条命令的退出状态码 ($? 为 0 代表成功)
            if [ $? -eq 0 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S'): [成功] $cmd 执行完毕！"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S'): [失败] $cmd 执行报错！"
                echo "任务链已中断，请查看日志 $log_file 排查问题。"
                exit 1  # 发生错误，立即退出整个脚本，不再执行后面的命令
            fi
        done
        
        echo "==================================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S'): 所有任务已按顺序全部顺利执行完毕！"
        exit 0 # 全部成功执行后，退出脚本
        
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S'): GPU 当前忙碌 (显存占用较高)，等待 10 分钟后再次检测..."
        sleep 600 # 暂停 600 秒 (10 分钟)
    fi
done
