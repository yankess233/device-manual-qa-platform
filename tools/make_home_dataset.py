"""生成家电域评测集（美的空调 40 页手册）→ eval/dataset-home.jsonl。

证据句全部逐字取自导入后的切片内容；生成后用 `python -m eval.link_gold` 校验，
解析失败的题会被点名，再回来改。
"""

from __future__ import annotations

import json
from pathlib import Path

ITEM = "美的 KFR-51LW/N8KS1-1P 变频分体落地式空调器"

BASE = {
    "history": [],
    "rewrite": "",
    "expect_item": ITEM,
    "needs_image": False,
    "difficulty": "easy",
    "expect_abstain": False,
}


def q(id_, type_, question, reference, evidence, keywords, sections=None, **kw):
    item = {
        "id": id_,
        "type": type_,
        "question": question,
        "reference": reference,
        "gold_sections": sections or [],
        "evidence": evidence,
        "answer_keywords": keywords,
        **BASE,
    }
    item.update(kw)
    return item


ITEMS = [
    # ---------------- fact 8 ----------------
    q("F01", "fact", "51 机的建议制冷面积是多少？", "23-34 平方米。", "23-34m", ["23-34"]),
    q("F02", "fact", "这款空调室内机噪声最低是多少分贝？", "室内机低档 22 dB(A)。", "22-38-42", ["22"]),
    q("F03", "fact", "51 机使用的制冷剂种类和充注量是多少？", "R32，1120g。", "R32/1120g", ["R32", "1120"]),
    q("F04", "fact", "遥控器可调节的温度范围是多少？", "16~30.0℃。", "温度可调范围是16~30.0℃", ["16", "30"]),
    q("F05", "fact", "遥控器定时最长可以设置多少小时？", "0-12 小时。", "可设定时间范围为0-12小时", ["12"]),
    q("F06", "fact", "睡眠功能开启后多久退出？", "8 小时后退出。", "8小时后退出", ["8"]),
    q("F07", "fact", "智清洁功能一般运行多久后自动退出？", "20-130 分钟后自动退出并关机。", "运行20-130分钟后自动退出智清洁运行并关机", ["20", "130"]),
    q("F08", "fact", "本机使用的制冷剂燃烧等级是什么？", "A2L，属于可燃性制冷剂。", "燃烧等级为A2L", ["A2L"]),
    # ---------------- table 7 ----------------
    q("T01", "table", "51 机的 HSPF 值是多少？", "3.71。", "KFR-51LW/N8KS1-1P为3.71", ["3.71"]),
    q("T02", "table", "51 机的 APF 值是多少？", "4.80。", "KFR-51LW/N8KS1-1P为4.80", ["4.80"]),
    q("T03", "table", "51 机的能效等级是几级？", "1 级。", "整机为能效等级，KFR-51LW/N8KS1-1P为1级", ["1级"]),
    q("T04", "table", "51 机的低温制热量是多少瓦？", "6280W。", "6280", ["6280"]),
    q("T05", "table", "72 机的建议制热面积是多少？", "36-49 平方米。", "36-49m", ["36-49"]),
    q("T06", "table", "51 机的中间制冷量和中间制热量分别是多少？", "2550/3600 W。", "KFR-51LW/N8KS1-1P为2550/3600", ["2550", "3600"]),
    q("T07", "table", "72 机的室内机噪声范围是多少？", "22-42-47 dB(A)。", "KFR-72LW/N8KS1-1P为22-42-47", ["22-42-47"]),
    # ---------------- procedure 7 ----------------
    q("P01", "procedure", "怎么清洁空调滤尘网？", "断电后抓住机身背面扣手位取出滤尘网，清水洗净，阴凉处晾干后装回。", "抓住机身背面的滤尘网扣手位", ["滤尘网", "晾干"], sections=["## 维护保养"]),
    q("P02", "procedure", "滤尘网建议多久清洁一次？", "建议每月清洁 1 次，灰尘多的环境要增加次数。", "建议每月清洁1次滤尘网", ["每月", "1次"]),
    q("P03", "procedure", "显示屏出现 E1 这类故障代码时应该怎么处理？", "关机断电，等 10 分钟后重新上电开机；仍显示代码则联系检修。", "关机并关闭电源 — 10分钟后打开电源并开机", ["10分钟", "检修"]),
    q("P04", "procedure", "长期停用空调前需要做哪些事？", "清洁机身与滤尘网，并检查排水管是否通畅。", "检查排水管是否通畅", ["排水管", "清洁"]),
    q("P05", "procedure", "怎么开启智清洁功能？", "按功能键后用方向键选择智清洁，再按确定键。", "按“功能”键后,按“^”“√”键选择智清洁,再按“确定”键", ["功能", "确定"]),
    q("P06", "procedure", "怎么设置定时关机？", "开机状态下按定时键，用方向键设定时间，范围为 0-12 小时。", "按\"定时\"键，遥控器定时图标和时间显示点亮", ["定时", "12"]),
    q("P07", "procedure", "防直吹功能开启后会怎样？", "自动调整上下导风板角度和风速。", "防直吹功能开启后，空调器自动调整上下导风板角度和风速", ["导风板", "风速"]),
    # ---------------- multi_turn 5 ----------------
    q("M01", "multi_turn", "那 72 机的制冷面积呢？", "32-48 平方米。", "32-48m", ["32-48"],
      history=[{"role": "user", "content": "51 机的建议制冷面积是多少？"},
               {"role": "assistant", "content": "51 机建议制冷面积为 23-34 平方米。"}],
      rewrite="美的 KFR-51LW/N8KS1-1P 空调 72 机的建议制冷面积是多少"),
    q("M02", "multi_turn", "它多久清洁一次比较合适？", "建议每月清洁 1 次。", "建议每月清洁1次滤尘网", ["每月"],
      history=[{"role": "user", "content": "滤尘网怎么拆下来？"},
               {"role": "assistant", "content": "抓住机身背面的滤尘网扣手位，向外用力拉并取出滤尘网。"}],
      rewrite="美的 KFR-51LW/N8KS1-1P 空调滤尘网多久清洁一次"),
    q("M03", "multi_turn", "这个模式下温度能调吗？", "送风模式下温度不可调。", "送风模式下温度不可调", ["送风", "不可调"],
      history=[{"role": "user", "content": "遥控器怎么切换模式？"},
               {"role": "assistant", "content": "依次按模式键，可在自动、制冷、抽湿、制热、送风之间切换。"}],
      rewrite="美的 KFR-51LW/N8KS1-1P 空调在送风模式下温度可以调节吗"),
    q("M04", "multi_turn", "那它运行的时候会很吵吗？", "室内机噪声 22-38-42 dB(A)，低档较安静。", "22-38-42", ["22"],
      history=[{"role": "user", "content": "这款空调是几匹的？"},
               {"role": "assistant", "content": "51 机属于 2 匹机型，建议制冷面积 23-34 平方米。"}],
      rewrite="美的 KFR-51LW/N8KS1-1P 空调运行时室内机噪声有多大"),
    q("M05", "multi_turn", "那停电以后还要重新设吗？", "断电后原定时设置自动取消，需重新设置。", "如在到达预约时间前发生断电，原有定时设置会自动取消", ["重新设置", "定时"],
      history=[{"role": "user", "content": "定时关机怎么设？"},
               {"role": "assistant", "content": "按定时键，用方向键设定 0-12 小时的时间即可。"}],
      rewrite="美的 KFR-51LW/N8KS1-1P 空调断电后定时设置是否需要重新设置"),
    # ---------------- out_of_scope 3 ----------------
    q("O01", "out_of_scope", "这台空调的整机保修期是几年？", "手册未给出保修年限，需查售后政策。", "", ["保修"],
      expect_abstain=True, difficulty="hard"),
    q("O02", "out_of_scope", "怎么把空调连到米家 APP？", "手册未涉及米家 APP 配网。", "", ["米家"],
      expect_abstain=True, difficulty="hard"),
    q("O03", "out_of_scope", "遥控器用的是几号电池？", "手册未标注遥控器电池型号。", "", ["电池"],
      expect_abstain=True, difficulty="hard"),
]


def main() -> None:
    out = Path("eval/dataset-home.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for item in ITEMS:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"写入 {out}：{len(ITEMS)} 题")


if __name__ == "__main__":
    main()