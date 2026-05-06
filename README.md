# dataopt

这是一个用于投资组合优化与回测的课程/实验项目。主入口是 [src/run_all.py](src/run_all.py)，会依次完成数据处理、参数估计、优化求解、回测、作图和结果导出。
PDF 报告位于 report/portfolio_report.pdf。

## 运行方式


只需要在项目根目录运行：

```bash
cd src
python run_all.py
```

如果你已经有本地价格数据，不想重新下载，可以用：

```bash
cd src
python run_all.py --skip-download
```


## 说明

- 已有示例数据放在 [data](data) 目录。
- 运行完成后的结果会输出到 [results](results)、[data](data) 和 [report](report) 目录。
