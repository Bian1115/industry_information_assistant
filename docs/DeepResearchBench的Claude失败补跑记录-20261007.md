# DeepResearchBench的Claude失败补跑记录

补跑日期：2026-10-07；开始：2026-10-07T20:06:00.087882+08:00；最近更新：2026-10-08T09:39:18.155264+08:00。
状态：补跑中；已核验有效评分 54/60；本补跑新增 11 次。

## 数据和配置口径
30题、758条官方标准，每题Claude两次。逐题完整正文、官方细则、模型claude-opus-5、服务默认采样和12000输出上限均沿用冻结配置。原助手分数不修改且不提供给裁判。
原进程运行时只重试已宣告失败的调用，原进程结束后才补齐其他缺失调用。补跑单并发；与原Claude进程重叠时总请求并发最多2。Qwen续跑由已有进程负责。

## 原始结果及核查
原始响应目录：/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/裁判原始记录；失败尝试：/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/失败尝试。
[补跑机器记录](/Users/bxy/projects/industry_information_assistant/backend/eval/data/DeepResearchBench参考报告30题双裁判一致性-20261007/Claude失败补跑状态.json)
有效评分要求流完整结束、全部官方细则齐全、分数在0–10之间、理由非空，且原始分数与格式处理后的分数逐项一致。只取有效响应，失败或缺失不计零分。

## 本次完成与失败
完成调用：题10-细则-claude-1、题38-细则-claude-1、题91-细则-claude-2、题73-细则-claude-1、题73-细则-claude-2、题14-细则-claude-2、题61-细则-claude-1、题1-细则-claude-1、题23-细则-claude-1、题23-细则-claude-2、题47-细则-claude-2
最近失败：题61-细则-claude-2：题61-细则-claude-2 调用失败 ['ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'ReadError:[Errno 54] Connection reset by peer', 'ReadError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out']；题61-细则-claude-2：题61-细则-claude-2 调用失败 ['ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'ReadError:[Errno 54] Connection reset by peer', 'ReadError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则缺失或重复', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out']；题61-细则-claude-2：题61-细则-claude-2 调用失败 ['ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'ReadError:[Errno 54] Connection reset by peer', 'ReadError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则缺失或重复', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'AssertionError:细则字段错误', 'AssertionError:细则缺失或重复', 'ReadError:[Errno 54] Connection reset by peer']；题22-细则-claude-2：题22-细则-claude-2 调用失败 ['AssertionError:细则字段错误', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out']；题22-细则-claude-2：题22-细则-claude-2 调用失败 ['AssertionError:细则字段错误', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误']；题22-细则-claude-2：题22-细则-claude-2 调用失败 ['AssertionError:细则字段错误', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误', 'AssertionError:细则字段错误', 'ReadTimeout:The read operation timed out', 'AssertionError:细则字段错误']；题23-细则-claude-1：题23-细则-claude-1 调用失败 ['ReadTimeout:流接收超过600秒总时限', 'AssertionError:细则字段错误', 'ConnectTimeout:_ssl.c:1003: The handshake operation timed out', 'ReadTimeout:流接收超过600秒总时限', 'ConnectError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out']；题31-细则-claude-1：题31-细则-claude-1 调用失败 ['ReadError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'ReadTimeout:The read operation timed out', 'ReadError:[Errno 54] Connection reset by peer', 'ReadTimeout:The read operation timed out']

## 局限性与结论
已测量：上述有效响应及其完整性；尚未评估：未补齐部分的一致性。结果不是人类金标，未做外部事实核验。
[完整评分与一致性报告](/Users/bxy/projects/industry_information_assistant/docs/DeepResearchBench参考报告30题评分一致性与分歧分析-20261007.md)
所有补齐后由页面生成器合并有效原始记录，更新每题HTML和一致性；未完成时保留真实缺失及失败原因。
