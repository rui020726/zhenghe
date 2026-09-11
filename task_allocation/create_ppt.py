#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成面向导师汇报的学术风格PPT — Deadline机制重构与测试分析"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── 颜色方案 (Nature-style: 白底 + 深蓝主色 + 珊瑚强调) ──
DARK    = RGBColor(0x22, 0x22, 0x22)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT  = RGBColor(0x1A, 0x56, 0xDB)   # 深蓝
CORAL   = RGBColor(0xE0, 0x4F, 0x3D)   # 红（强调）
GREEN   = RGBColor(0x2E, 0x8B, 0x57)   # 绿（正面）
GRAY    = RGBColor(0x88, 0x88, 0x88)
LIGHT_BG = RGBColor(0xF5, 0xF7, 0xFA)  # 浅灰蓝背景
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

# ── 辅助函数 ──
def add_blank_slide():
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=DARK, alignment=PP_ALIGN.LEFT, font_name="Microsoft YaHei"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf

def add_bullet_frame(slide, left, top, width, height, bullets, font_size=15,
                     color=DARK, spacing=Pt(8), font_name="Microsoft YaHei"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (level, text) in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.level = level
        p.space_after = spacing
        if level == 0:
            p.font.bold = False
    return tf

def add_line(slide, left, top, width, color=ACCENT, thickness=Pt(2)):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Pt(2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_rect(slide, left, top, width, height, color=LIGHT_BG):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_metric_card(slide, left, top, label, value, value_color=DARK):
    """一个指标卡片"""
    w, h = 2.8, 1.4
    add_rect(slide, left, top, w, h, LIGHT_BG)
    add_textbox(slide, left + 0.15, top + 0.15, w - 0.3, 0.5, label,
                font_size=12, color=GRAY)
    add_textbox(slide, left + 0.15, top + 0.55, w - 0.3, 0.7, value,
                font_size=30, bold=True, color=value_color)

def add_slide_number(slide, num):
    add_textbox(slide, 12.2, 7.05, 0.8, 0.35, str(num),
                font_size=10, color=GRAY, alignment=PP_ALIGN.RIGHT)

def add_section_header(slide, title, subtitle=None):
    """页面顶部统一标题栏"""
    add_line(slide, 0.6, 0.65, 12.1, ACCENT, Pt(3))
    add_textbox(slide, 0.6, 0.2, 10, 0.45, title, font_size=26, bold=True)
    if subtitle:
        add_textbox(slide, 0.6, 0.72, 10, 0.35, subtitle, font_size=13, color=GRAY)

# ═══════════════════════════════════════════
# Slide 1 — 标题页
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_rect(slide, 0, 0, 13.333, 7.5, RGBColor(0x0D, 0x1B, 0x36))  # 深色背景
add_line(slide, 0.8, 3.0, 3.5, CORAL, Pt(3))
add_textbox(slide, 0.8, 1.8, 11, 1.2,
            "CTDE范式下异构MRTA系统",
            font_size=40, bold=True, color=WHITE)
add_textbox(slide, 0.8, 3.2, 11, 0.7,
            "Deadline机制重构与测试分析  ·  本周工作汇报",
            font_size=20, color=RGBColor(0xBB, 0xCC, 0xDD))
add_textbox(slide, 0.8, 5.5, 5, 0.4, "2026年5月18日", font_size=14, color=GRAY)
add_textbox(slide, 0.8, 5.9, 5, 0.4, "指导教师: ______", font_size=14, color=GRAY)

# ═══════════════════════════════════════════
# Slide 2 — 问题发现: 测试结果异常
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "问题发现：测试结果异常",
                   "3种Agent规模 × 5种任务负载，15组共1500 episodes")

# 四个指标卡片
add_metric_card(slide, 0.6, 1.3, "Deadline 成功率", "10% – 22%", CORAL)
add_metric_card(slide, 3.7, 1.3, "任务完成率", "94%", GREEN)
add_metric_card(slide, 6.8, 1.3, "任务过期率", "71% – 85%", CORAL)
add_metric_card(slide, 9.9, 1.3, "资源利用率", "15% – 17%", CORAL)

# 矛盾点
add_rect(slide, 0.6, 3.3, 12.1, 1.8)
add_textbox(slide, 0.9, 3.45, 11.5, 0.4, "⚠ 核心矛盾", font_size=18, bold=True, color=CORAL)
add_bullet_frame(slide, 0.9, 3.9, 11.5, 1.2, [
    (0, "• 模型能把 94% 的任务做完，说明任务执行能力没有问题"),
    (0, "• 但 71%–85% 的任务都过期了，deadline 成功率仅 10%–22%"),
    (0, "• 模型学会了"完成任务"，但没有学会"赶在 deadline 前完成任务""),
], font_size=14)

# 问题引出
add_textbox(slide, 0.6, 5.5, 12, 0.6,
            "→ 需要回答：为什么模型不关心 deadline？奖励信号出了什么问题？",
            font_size=16, bold=True, color=ACCENT)
add_slide_number(slide, 1)

# ═══════════════════════════════════════════
# Slide 3 — 根因定位: 奖励机制缺陷
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "根因定位：终端奖励计算存在结构性缺陷")

# 左: 问题说明
add_rect(slide, 0.6, 1.3, 6.0, 4.5)
add_textbox(slide, 0.9, 1.5, 5.4, 0.4, "缺陷机制", font_size=18, bold=True, color=CORAL)
add_bullet_frame(slide, 0.9, 2.1, 5.4, 3.5, [
    (0, "1. 过期任务自动标记 finished = True"),
    (0, "    → 任务过期 = 任务完成，语义混淆"),
    (0, "2. 任务完成奖励不区分"过期"和"真正完成""),
    (0, "    → 过期任务拿到 +200 完成奖励"),
    (0, "3. 过期惩罚仅 -80，不足以抵销"),
    (0, "    → 净收益 = +200 − 80 = +120 (正值！)"),
    (0, "4. 过期任务立即从 agent 观测中消失"),
    (0, "    → 没有持续的追赶压力"),
], font_size=13)

# 右: 收益对比
add_rect(slide, 7.0, 1.3, 5.7, 4.5)
add_textbox(slide, 7.3, 1.5, 5.1, 0.4, "修复前：各状态净收益", font_size=18, bold=True, color=ACCENT)

# 修复前表格
table_data = [
    ("状态", "完成奖励", "过期惩罚", "净收益"),
    ("按时完成", "+200", "0", "+240 ✓"),
    ("过期 (模型选择)", "+200", "−80", "+120 ← 更划算"),
    ("未触碰", "0", "0", "0"),
]
table_left = Inches(7.3)
table_top = Inches(2.1)
rows, cols = len(table_data), len(table_data[0])
table = slide.shapes.add_table(rows, cols, Inches(4.8), Inches(2.2), table_left, table_top).table
table.columns[0].width = Inches(1.5)
table.columns[1].width = Inches(1.1)
table.columns[2].width = Inches(1.1)
table.columns[3].width = Inches(1.1)

for r in range(rows):
    for c in range(cols):
        cell = table.cell(r, c)
        cell.text = table_data[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.name = "Microsoft YaHei"
            p.alignment = PP_ALIGN.CENTER
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
            elif r == 2:
                p.font.color.rgb = CORAL
                p.font.bold = True

# 结论
add_textbox(slide, 7.3, 4.6, 5.1, 1.0,
            "结论：模型学到的"最优策略"是不赶 deadline，\n"
            "等任务自动过期反而净赚 +120 分。\n"
            "这解释了三项指标的同时异常。",
            font_size=13, color=CORAL)
add_slide_number(slide, 2)

# ═══════════════════════════════════════════
# Slide 4 — 解决方案: Deadline 机制重构
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "解决方案：Deadline 机制重构 —— 硬约束 → 软约束",
                   "借鉴 NeurIPS 2025 TARGETNET 论文的软约束设计理念")

# 三个改动方向
items = [
    ("过期检测", "过期后任务不消失、不释放成员、仍可被选择\n→ Agent 可以继续执行过期任务，完成"补救"",
     RGBColor(0x1A, 0x56, 0xDB)),
    ("奖励重构", "区分四种状态，形成清晰的奖励梯度\n按时完成+240 / 过期后完成+20 / 未触碰-40 / 过期放弃-120",
     RGBColor(0x2E, 0x8B, 0x57)),
    ("观测增强", "Agent 观测中增加线性剩余时间信号\n→ 双通道时间感知：tanh + 线性倒计时",
     RGBColor(0x8B, 0x45, 0x13)),
]
for i, (title, desc, clr) in enumerate(items):
    x = 0.6 + i * 4.1
    add_rect(slide, x, 1.3, 3.8, 2.5)
    add_textbox(slide, x + 0.25, 1.45, 3.3, 0.4, f"▎{title}", font_size=20, bold=True, color=clr)
    add_textbox(slide, x + 0.25, 2.0, 3.3, 1.6, desc, font_size=13, color=DARK)

# 奖励梯度表
add_textbox(slide, 0.6, 4.2, 12, 0.4, "修复后：四种状态净收益梯度", font_size=16, bold=True, color=DARK)
table_data2 = [
    ("按时完成", "+200", "—", "—", "+240 ✓", GREEN),
    ("过期后完成", "+100 (×0.5)", "−80", "—", "+20", DARK),
    ("从未触碰", "—", "—", "−40", "−40", ACCENT),
    ("过期放弃", "—", "−80", "−40", "−120", CORAL),
]
t2 = slide.shapes.add_table(5, 5, Inches(0.6), Inches(4.7), Inches(12.1), Inches(2.3)).table
t2.columns[0].width = Inches(2.4)
t2.columns[1].width = Inches(2.4)
t2.columns[2].width = Inches(2.4)
t2.columns[3].width = Inches(2.4)
t2.columns[4].width = Inches(2.5)

for r in range(5):
    for c in range(5):
        cell = t2.cell(r, c)
        if r == 0:
            cell.text = ["", "完成奖励", "过期惩罚", "未完成惩罚", "净收益"][c]
        else:
            cell.text = table_data2[r-1][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.name = "Microsoft YaHei"
            p.alignment = PP_ALIGN.CENTER
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
            elif c == 4:
                p.font.bold = True
                p.font.color.rgb = table_data2[r-1][5]

add_textbox(slide, 0.6, 7.05, 10, 0.35,
            "梯度清晰：按时完成 >> 过期补救 > 不理睬 > 过期放弃",
            font_size=13, bold=True, color=ACCENT)
add_slide_number(slide, 3)

# ═══════════════════════════════════════════
# Slide 5 — 论文调研
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "参考论文调研：Deadline 处理机制对比")

# 对比表
t3_data = [
    ["", "GMATANN", "SHPPO", "TARGETNET"],
    ["Deadline 概念", "无", "无", "有 ✓（时间窗约束）"],
    ["过期任务处理", "—", "—", "不删除，标记不可行"],
    ["约束性质", "—", "—", "软约束（扣分不删除）"],
    ["决策范式", "序贯决策", "序贯决策", "一次性调度"],
    ["适用场景", "静态任务", "战斗/游戏", "规划调度"],
]
t3 = slide.shapes.add_table(6, 4, Inches(0.6), Inches(1.2), Inches(12.1), Inches(3.2)).table
t3.columns[0].width = Inches(3.0)
t3.columns[1].width = Inches(3.0)
t3.columns[2].width = Inches(3.0)
t3.columns[3].width = Inches(3.1)
for r in range(6):
    for c in range(4):
        cell = t3.cell(r, c)
        cell.text = t3_data[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.name = "Microsoft YaHei"
            p.alignment = PP_ALIGN.CENTER
            if r == 0 or c == 0:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
                p.font.color.rgb = WHITE
            if c == 3 and r in [1, 2, 3]:
                p.font.color.rgb = GREEN
                p.font.bold = True

# 关键发现
add_rect(slide, 0.6, 4.8, 12.1, 1.6)
add_textbox(slide, 0.9, 4.95, 11.5, 0.4, "💡 关键发现", font_size=16, bold=True, color=ACCENT)
add_bullet_frame(slide, 0.9, 5.3, 11.5, 0.9, [
    (0, "• TARGETNET 是唯一显式建模 deadline 的参考论文，采用软约束设计"),
    (0, "• 但其决策范式为"一次性规划器"（图编码 → 单次前向 → 完整调度），与我们的序贯决策场景不同"),
    (0, "• 我们的软约束方案在理念上对齐 TARGETNET，但在实现上适配序贯决策的特点"),
], font_size=13)
add_slide_number(slide, 4)

# ═══════════════════════════════════════════
# Slide 6 — 训练参数与Curriculum分析
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "Curriculum 设计与 Deadline Tightness 分析")

# 左: curriculum 阶段
add_textbox(slide, 0.6, 1.3, 5.5, 0.4, "▎5 阶段 Curriculum：Deadline Tightness 逐阶段收紧",
            font_size=18, bold=True)
t4_data = [
    ["阶段", "Tightness 范围", "Slack 程度"],
    ["Stage 1", "1.0 – 2.5", "宽松"],
    ["Stage 2", "0.7 – 2.0", "中等"],
    ["Stage 3", "0.5 – 1.5", "偏紧"],
    ["Stage 4", "0.3 – 1.2", "紧"],
    ["Stage 5", "0.3 – 1.0", "极紧（几乎无 slack）"],
]
t4 = slide.shapes.add_table(6, 3, Inches(0.6), Inches(1.9), Inches(5.5), Inches(2.8)).table
t4.columns[0].width = Inches(1.5)
t4.columns[1].width = Inches(2.0)
t4.columns[2].width = Inches(2.0)
for r in range(6):
    for c in range(3):
        cell = t4.cell(r, c)
        cell.text = t4_data[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.name = "Microsoft YaHei"
            p.alignment = PP_ALIGN.CENTER
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
            if r == 5:
                p.font.color.rgb = CORAL

# 右: 发现
add_rect(slide, 6.6, 1.3, 6.1, 3.8)
add_textbox(slide, 6.9, 1.5, 5.5, 0.4, "⚙ 发现的问题", font_size=18, bold=True, color=CORAL)
add_bullet_frame(slide, 6.9, 2.1, 5.5, 2.8, [
    (0, "• Tightness < 1.0 时保底机制主导："),
    (0, "  deadline = max(estimated × tightness,"),
    (0, "                   total_work × 1.1)"),
    (0, "  后期大部分任务 deadline 卡在"),
    (0, "  total_work × 1.1 的保底值"),
    (0, ""),
    (0, "• 后果：Stage 4-5 随机性退化，"),
    (0, "  各任务 deadline 趋于统一"),
    (0, ""),
    (0, "• 建议：调整下限保持 slack 空间，"),
    (0, "  避免保底机制在高阶段主导"),
], font_size=12)
add_slide_number(slide, 5)

# ═══════════════════════════════════════════
# Slide 7 — 后续计划
# ═══════════════════════════════════════════
slide = add_blank_slide()
add_section_header(slide, "后续工作计划")

plans = [
    ("1", "重新训练模型",
     "用修复后的软约束机制重新训练，预期 deadline 成功率显著提升",
     ACCENT),
    ("2", "修复前后对比验证",
     "定量比较 deadline 成功率、过期率、资源利用率的变化，验证方案有效性",
     GREEN),
    ("3", "可行性辅助预测头",
     "受 TARGETNET 启发，Actor 增加二分类头预测"能否在 deadline 前完成"，让模型显式学习可行性判断",
     RGBColor(0x8B, 0x45, 0x13)),
    ("4", "信用分配优化",
     "探索 COMA-style counterfactual baseline，解决多阶段任务中 agent 贡献难以区分的问题",
     RGBColor(0x6A, 0x5A, 0xCD)),
]

for i, (num, title, desc, clr) in enumerate(plans):
    y = 1.3 + i * 1.45
    add_rect(slide, 0.6, y, 12.1, 1.2)
    add_textbox(slide, 0.85, y + 0.1, 0.5, 0.6, num, font_size=28, bold=True, color=clr)
    add_textbox(slide, 1.5, y + 0.12, 10.5, 0.4, title, font_size=18, bold=True, color=DARK)
    add_textbox(slide, 1.5, y + 0.55, 10.5, 0.5, desc, font_size=13, color=GRAY)

add_slide_number(slide, 6)

# ── 保存 ──
output_path = "weekly_report.pptx"
prs.save(output_path)
print(f"✅ PPT 已生成: {output_path}")
print(f"   共 {len(prs.slides)} 页")
