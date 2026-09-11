#!/bin/bash
# 训练监控脚本：每2小时检查训练是否结束，结束后自动运行测试
# 用法: bash monitor.sh <训练日志路径>

PROJECT_DIR="/d/code/task-03-修改为CTDE范式"
PYTHON="/d/project_env/python.exe"
TEST_SCRIPT="old_test.py"
CHECK_INTERVAL=7200  # 2小时
TRAIN_LOG="${1:-}"

echo "============================================"
echo "[Monitor] 开始监控训练进程"
echo "[Monitor] 检查间隔: ${CHECK_INTERVAL}s (2小时)"
echo "[Monitor] 项目目录: ${PROJECT_DIR}"
echo "[Monitor] 训练日志: ${TRAIN_LOG:-未指定}"
echo "[Monitor] 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

get_episode_info() {
    if [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ]; then
        # 从训练日志提取最新 episode 信息 (basic regex, no -P)
        local last_line=$(grep 'DL_SR:' "$TRAIN_LOG" 2>/dev/null | tail -1)
        if [ -n "$last_line" ]; then
            # 提取 Ep XXX (紧跟 Ep 后的数字)
            local last_ep=$(echo "$last_line" | sed -n 's/.*Ep \([0-9]\+\).*/\1/p')
            # 提取 | R:XXX.X | (匹配竖线空格R:的格式，避免匹配到DL_SR)
            local last_r=$(echo "$last_line" | sed -n 's/.*| R:\([0-9.-]\+\).*/\1/p')
            # 提取 DL_SR:XX.XX%
            local last_dl=$(echo "$last_line" | sed -n 's/.*DL_SR:\([0-9.]\+%\).*/\1/p')
            echo "  当前回合: Ep ${last_ep:-?} | R:${last_r:-?} | DL_SR:${last_dl:-?}"
        else
            echo "  当前回合: (无法解析)"
        fi
    fi
}

TEST_TRIGGERED=false

while true; do
    RUNNING=$(tasklist 2>/dev/null | grep -c "python.exe" || echo "0")
    RUNNING=$(echo "$RUNNING" | tr -d '\r\n ')

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [ -z "$RUNNING" ] || [ "$RUNNING" = "0" ]; then
        if [ "$TEST_TRIGGERED" = false ]; then
            echo ""
            echo "[${TIMESTAMP}] 训练已结束，开始运行测试..."
            TEST_TRIGGERED=true

            RESULT_DIR="${PROJECT_DIR}/test_results/$(date '+%Y%m%d_%H%M%S')"
            mkdir -p "$RESULT_DIR"

            cd "$PROJECT_DIR"
            echo "[Monitor] 测试结果目录: ${RESULT_DIR}"
            echo ""

            "$PYTHON" "$TEST_SCRIPT" 2>&1 | tee "${RESULT_DIR}/test_log.txt"
            TEST_EXIT_CODE=$?

            echo ""
            if [ $TEST_EXIT_CODE -eq 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 测试完成"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 测试退出，退出码: ${TEST_EXIT_CODE}"
            fi

            LATEST_JSON=$(ls -t "$PROJECT_DIR/test_results"/*.json 2>/dev/null | head -1)
            if [ -n "$LATEST_JSON" ] && [ -f "$LATEST_JSON" ]; then
                cp "$LATEST_JSON" "$RESULT_DIR/"
                echo "[Monitor] 已复制结果文件: $(basename "$LATEST_JSON")"
            fi

            if [ -d "$PROJECT_DIR/test_results/figures" ]; then
                FIGURE_COUNT=$(ls "$PROJECT_DIR/test_results/figures" 2>/dev/null | wc -l)
                if [ "$FIGURE_COUNT" -gt 0 ]; then
                    mkdir -p "$RESULT_DIR/figures"
                    cp "$PROJECT_DIR/test_results/figures"/* "$RESULT_DIR/figures/" 2>/dev/null
                    echo "[Monitor] 已复制 ${FIGURE_COUNT} 个图表文件"
                fi
            fi

            echo ""
            echo "[Monitor] 结果已保存到: ${RESULT_DIR}"
            ls -la "$RESULT_DIR/"
            echo ""
            echo "[Monitor] 监控任务完成，退出。"
            exit 0
        else
            echo "[${TIMESTAMP}] 训练已结束（测试已执行），退出。"
            exit 0
        fi
    else
        echo "[${TIMESTAMP}] 训练进行中 | 进程数: ${RUNNING}"
        get_episode_info
        echo "  ${CHECK_INTERVAL}s 后再次检查..."
        sleep $CHECK_INTERVAL
    fi
done
