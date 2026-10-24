#!/usr/bin/env python3
"""本次特定柱状图的可复现图形估读，不当作通用OCR或原始精确数据。"""
import hashlib
import itertools
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / 'backend/eval/data/网页编码修复与三份资源补救验证-20261017/补充资源/49-page-007'


def main():
    source = FOLDER / '原始下载资源.png'
    image = Image.open(source).convert('RGB')
    assert image.size == (1178, 565)
    pixels = image.load()
    # 刻度主线像素位于y=425和99；按绘图线宽取近似中心425.5和99.5。
    baseline, upper_tick = 425.5, 99.5
    bars = []
    for gender, color in [('男', (0, 55, 120)), ('女', (190, 190, 190))]:
        columns = [x for x in range(138, 1113) if sum(pixels[x,y]==color for y in range(100,425)) >= 12]
        groups = [[t[1] for t in group] for _,group in itertools.groupby(enumerate(columns), lambda t:t[1]-t[0])]
        assert len(groups) == 17
        for index, group in enumerate(groups):
            tops = sorted(min(y for y in range(100,425) if pixels[x,y]==color) for x in group)
            top = tops[len(tops)//2] - 0.5
            interpolated = (baseline-top)/(baseline-upper_tick)*7000
            bars.append({'gender':gender, 'index':index, 'x_min':min(group), 'x_max':max(group),
                         'top_y':top, 'interpolated_10k':round(interpolated,2),
                         'approx_population_10k':round(interpolated/100)*100})
    measurement = {'method':'17组男女柱按图例纯色像素定位，依据原图纵轴刻度线性换算，取整至100万人。年龄顺序、图例、标题和来源另经原图视觉核对。',
                   'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'baseline_y':baseline, 'axis_7000_y':upper_tick,
                   'precision':'图形近似估读；取整至100万人，像素/绘图分辨率误差未做独立金标评测，非统计原始精确数据。', 'bars':bars}
    (FOLDER/'柱形高度测量.json').write_text(json.dumps(measurement,ensure_ascii=False,indent=2)+'\n')
    ages=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+']
    rows=[{'age':age, 'male_approx_10k':bars[i]['approx_population_10k'], 'female_approx_10k':bars[17+i]['approx_population_10k']} for i,age in enumerate(ages)]
    (FOLDER/'图表估读数据.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    text='图表18：2020年中国人口结构（分性别、年龄）\n\n单位：万人。深蓝色为男，灰色为女。\n以下为从原始PNG柱高与纵轴刻度估读的近似值，取整至100万人，不是图中标注的精确数据，不应用于精确计算。\n\n| 年龄 | 男（约，万人） | 女（约，万人） |\n| --- | ---: | ---: |\n'
    for row in rows:
        text+=f'| {row["age"]} | {row["male_approx_10k"]} | {row["female_approx_10k"]} |\n'
    text+='\n原图以红色虚线框标出20-39岁区间。'
    for index in [6,10]:
        row=rows[index]
        text+=f'图形中{row["age"]}岁组男性约{row["male_approx_10k"]}万人、女性约{row["female_approx_10k"]}万人。'
    text+='\n\n图下注明资料来源：World Population Statistics，华泰证券研究所。\n'
    (FOLDER/'待评正文.txt').write_text(text)
    (FOLDER/'提取记录.json').write_text(json.dumps({'page_id':'49-page-007','extractor':'macOS Vision文字识别＋针对原图的柱高估读＋视觉核对','scope':'仅此图的小样本资源验证，尚未接入通用网页读取器','bars':34,'age_groups':17,'evaluated_chars':len(text),'content_sha256':hashlib.sha256(text.encode()).hexdigest(),'estimated_not_exact':True},ensure_ascii=False,indent=2)+'\n')
    print(text)


if __name__=='__main__':
    main()
