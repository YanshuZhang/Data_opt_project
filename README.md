# Portfolio Optimization Project

本文件包完成作业中的必做部分：真实数据获取与清洗、QP/SOCP 商业求解器基准、ADMM、PDHG、真实数据回测、敏感性分析和失败案例说明。不包含 optional extension 和 bonus task。

## 1. 环境安装

建议使用新的 Python 环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

作业要求使用至少一个商业求解器。推荐 MOSEK。安装后需要配置 `mosek.lic`，并确认：

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

输出中应包含 `MOSEK` 或 `GUROBI`。

## 2. 运行完整实验

在项目根目录运行：

```bash
python src/run_all.py --compile-pdf
```

运行后会生成：

```text
data/                       下载和清洗后的数据
results/csv/                数值结果 CSV
results/tables/             LaTeX 表格
results/figures/            图像结果
report/portfolio_report.pdf 中文报告 PDF
```

如果还没有安装商业求解器，但想先检查代码流程，可以临时运行：

```bash
python src/run_all.py --allow-open-source-fallback --compile-pdf
```

这只用于调试。正式报告结果应使用 MOSEK 或 Gurobi。

## 3. 主要参数

主要实验参数在 `src/config.py` 中修改，包括资产池、时间区间、训练/验证/测试划分、调参网格、回测窗口长度和调仓频率。

默认资产池为 30 只高流动性美股，默认使用 Yahoo Finance 复权日频价格。

## 4. 报告

中文报告源码在：

```text
report/portfolio_report.tex
```

报告不粘贴代码，只说明任务目标、数学模型、算法实现、测试结果和分析。运行 `run_all.py` 后，结果表格和图片会自动写入报告。
