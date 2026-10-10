# 2026-10-03 历史核验归档

[返回资料首页](../../README.md) · [主框架的历史证据](../../RLVR工业流程.md#historical-evidence)

这是从前次临时工作目录找回并按原字节复制的历史材料。它们对应原五题，不对应新 Hot 100 最大子数组和实验。

- [audit.py](audit.py)：当时的五题 seed、小域/随机对照和受控 scorer 探针脚本。
- [audit.json](audit.json)：当时 Python3.14.6、7088组穷举加1000组随机对照、11条seed及受控分数结果。
- [original.md](original.md)：当时审核脚本读取的基线文档快照，含hash依据；不作为当前课程正文。它保留当时定界符和原位置的相对路径，不为当前导航或公式检查重写历史字节。
- [validation.json](validation.json)：当时文档完整性/链接/编码验证摘要，其文件hash与行数属于当时版本，不能当作当前目录的验证结果。

归档自身不随本课命令更新。**不要在这里直接运行 audit.py**：旧脚本会向自身目录写同名报告，可能覆盖旧证据；它依赖原 Windows 工作区路径和当时源码/数据。需要复查时先复制 audit.py 与 original.md 到一个新的工作目录，核对源码版本和输入hash，再运行副本并保留新stdout/报告。新核验单独标日期，不冒充原执行日志。

新演示的稳定运行入口与当前报告见 [maximum-subarray](../maximum-subarray/README.md)。两套报告各自说明数据、算法、backend与执行范围。找回脚本提高可复查性，不使有限对照变成全输入正确性证明，也不将受控subprocess升级成生产Sandbox。
