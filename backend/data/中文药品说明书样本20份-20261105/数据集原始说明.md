---
license: Apache License 2.0
tags:
- 药品说明书
- 医药数据
- PDF
- OCR
- 信息抽取
- pharmaceutical-instructions
- document-ai
configs:
- config_name: default
  data_files:
  - split: train
    path: metadata/drug_instructions_manifest.jsonl
---

# 药品说明书样本20份

本数据集是药品说明书 PDF 开源样本，包含 20 份药品说明书 PDF 原件及对应 JSON 元数据。每份样本以批准文号为文件名，并提供药品名称、剂型、生产企业、PDF 页数、文件大小和 SHA256 校验值等字段，便于检索、加载和二次处理。

本次发布保留 PDF 原件，并提供面向数据预览、检索和程序加载的 JSONL 清单。完整 PDF 位于 `data/pdfs/`，结构化元数据位于 `metadata/`。

如需了解更多药品说明书、药品基础库、药品 SKU、药品商品名、医药知识库建设、OCR 解析或批量授权合作，可发送邮件至 zhouhaoran@shujuyoupu.com。

## 数据集简介

- 数据类型：药品说明书 PDF、药品批准文号元数据、文件级校验信息。
- 样本数量：20 份药品说明书。
- PDF 总大小：约 3.20 MB。
- PDF 总页数：96 页。
- 样本组成：20 份 PDF 原件、1 个说明书清单、1 个文件清单、1 个数据集摘要。
- 覆盖范围：国产化学药品、中成药/港澳传统药、境外生产药品；剂型覆盖注射液、薄膜衣片、胶囊、片、粉针剂、大蜜丸、颗粒、糖衣片、水丸、薄膜衣缓释片、口服散、糖浆、口服溶液、水蜜丸、冻干粉针剂、滴眼液、软膏、软胶囊、双层片、粉针(静脉滴注)。
- 适用任务：药品说明书 OCR、版面解析、字段抽取、批准文号识别、药品实体识别、药品知识库构建、医学文档问答和检索增强评测。

## 数据结构

```text
open_data/modelscope/drug-instructions-sample-20/
  README.md
  data/
    pdfs/
      国药准字H20244807.pdf
      国药准字H20244371.pdf
      ...
  metadata/
    drug_instructions_manifest.jsonl
    files_manifest.jsonl
    dataset_summary.json
```

## 样本清单

| instruction_id | 批准文号 | 药品名称 | 剂型 | 批准文号类型 | PDF 页数 | PDF 大小 |
|---|---|---|---|---|---:|---:|
| instruction-001 | 国药准字H20244807 | 盐酸多巴酚丁胺注射液 | 注射液 | 国产化学药品 | 4 | 0.154 MB |
| instruction-002 | 国药准字H20244371 | 沙库巴曲缬沙坦钠片 | 薄膜衣片 | 国产化学药品 | 9 | 0.236 MB |
| instruction-003 | 国药准字H20230008 | 复方利血平氨苯蝶啶胶囊 | 胶囊 | 国产化学药品 | 3 | 0.095 MB |
| instruction-004 | 国药准字H20243946 | 富马酸比索洛尔片 | 片 | 国产化学药品 | 7 | 0.212 MB |
| instruction-005 | 国药准字H20244064 | 注射用盐酸罗沙替丁醋酸酯 | 粉针剂 | 国产化学药品 | 4 | 0.154 MB |
| instruction-006 | 国药准字Z15020095 | 通宣理肺丸 | 大蜜丸 | 中成药/港澳传统药 | 2 | 0.089 MB |
| instruction-007 | 国药准字H42022296 | 氯化钾颗粒 | 颗粒 | 国产化学药品 | 4 | 0.160 MB |
| instruction-008 | 国药准字H41025663 | 盐酸倍他司汀片 | 糖衣片 | 国产化学药品 | 3 | 0.133 MB |
| instruction-009 | 国药准字Z61020577 | 藿香正气丸 | 水丸 | 中成药/港澳传统药 | 2 | 0.089 MB |
| instruction-010 | 国药准字HJ20140780 | 琥珀酸美托洛尔缓释片 | 薄膜衣缓释片 | 境外生产药品 | 7 | 0.230 MB |
| instruction-011 | 国药准字H20249673 | 复方聚乙二醇电解质散（Ⅲ） | 口服散 | 国产化学药品 | 4 | 0.152 MB |
| instruction-012 | 国药准字H33022369 | 浓维磷糖浆 | 糖浆 | 国产化学药品 | 3 | 0.103 MB |
| instruction-013 | 国药准字H20253657 | 乳果糖口服溶液 | 口服溶液 | 国产化学药品 | 3 | 0.111 MB |
| instruction-014 | 国药准字Z33020085 | 润肠丸 | 水蜜丸 | 中成药/港澳传统药 | 2 | 0.074 MB |
| instruction-015 | 国药准字H22023692 | 注射用尿激酶 | 冻干粉针剂 | 国产化学药品 | 3 | 0.156 MB |
| instruction-016 | 国药准字H20247102 | 左氧氟沙星滴眼液 | 滴眼液 | 国产化学药品 | 4 | 0.173 MB |
| instruction-017 | 国药准字H20249414 | 他克莫司软膏 | 软膏 | 国产化学药品 | 10 | 0.253 MB |
| instruction-018 | 国药准字H20244162 | 氯苯唑酸葡胺软胶囊 | 软胶囊 | 国产化学药品 | 5 | 0.182 MB |
| instruction-019 | 国药准字H20244264 | 替米沙坦氨氯地平片 | 双层片 | 国产化学药品 | 12 | 0.264 MB |
| instruction-020 | 国药准字H20249595 | 注射用尼可地尔 | 粉针(静脉滴注) | 国产化学药品 | 5 | 0.176 MB |

## JSON 元数据

`metadata/drug_instructions_manifest.jsonl` 每行对应一份药品说明书：

```json
{
  "instruction_id": "instruction-001",
  "approval_number": "国药准字H20244807",
  "approval_type": "国产化学药品",
  "drug_name": "盐酸多巴酚丁胺注射液",
  "drug_name_en": null,
  "trade_name": [],
  "dosage_form": "注射液",
  "specification": null,
  "manufacturer": ["南京泽恒医药技术开发有限公司"],
  "pdf_path": "data/pdfs/国药准字H20244807.pdf",
  "pdf_file_name": "国药准字H20244807.pdf",
  "pdf_size_bytes": 161002,
  "pdf_size_mb": 0.154,
  "pdf_pages": 4,
  "pdf_sha256": "...",
  "text_extractable_by_pypdf": true
}
```

`metadata/files_manifest.jsonl` 每行对应一个文件，包含 `instruction_id`、`approval_number`、`drug_name`、`file_path`、`file_type`、`size_bytes`、`size_mb`、`sha256`。

`metadata/dataset_summary.json` 保存本次样本的文件数量、PDF 总大小、总页数、剂型分布、批准文号类型分布和元数据文件列表。

## 加载方式

通过 ModelScope SDK 加载结构化清单：

```python
from modelscope import MsDataset

ds = MsDataset.load(
    "drug-instructions-sample-20",
    namespace="SHPDataGR",
    subset_name="default",
    split="train",
    version="master",
)

print(ds[0]["drug_name"])
print(ds[0]["approval_number"])
print(ds[0]["pdf_path"])
```

如需读取 PDF，请根据 `pdf_path` 定位到 `data/pdfs/` 下的对应文件。

## 数据质量和限制

- 本数据集是 20 份 PDF 样本，不代表完整药品说明书库。
- PDF 原件可能存在扫描质量、文本层、字体嵌入、表格排版和换行差异。
- `text_sample` 由程序从 PDF 前两页尝试抽取，仅用于快速判断文本层可读性，不应替代 PDF 原文。
- 同一药品可能存在不同批准文号、规格、企业或说明书版本，本样本保留批准文号粒度。
- 药品说明书内容可能随监管要求和企业更新发生变化，请以最新官方批准说明书为准。

## 合规和使用说明

本数据集仅用于药品文档解析、信息抽取、检索、知识库建设、模型训练与评测等合规用途，不构成用药建议、诊疗建议或药品推荐。使用者如基于本数据进行研究、训练、产品开发或再发布，应遵守数据集许可证、平台规则以及药品信息合规要求。

## 合作与联系

本数据集为药品说明书数据样本。若需要更多药品说明书 PDF、结构化字段、药品基础库、药品 SKU、药品商品名、OCR 解析结果、私有化语料建设或模型训练数据合作，请发送邮件至 zhouhaoran@shujuyoupu.com。

## 引用方式

暂无固定论文引用格式。使用时建议注明数据集名称、版本、访问平台和数据批次。
