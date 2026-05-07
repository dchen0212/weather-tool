# MeteoAgent

一个面向任务的天气检索与分析 agent，支持 `Streamlit` 图形界面、命令行导出和轻量级自然语言任务执行。项目基于 **NASA POWER** 获取指定经纬度和时间范围的日尺度气象数据，并提供真实值与预测值 CSV 的误差分析与摘要洞察功能。  
A task-oriented weather retrieval and analysis agent that supports a `Streamlit` graphical interface, command-line export, and lightweight natural-language task execution. It uses **NASA POWER** to retrieve daily weather data for a given latitude/longitude and date range, and provides real-vs-predicted CSV evaluation plus summary insights.

仓库地址 / Repository: [dchen0212/weather-tool](https://github.com/dchen0212/weather-tool)

## 功能简介 | Features

- 根据经纬度和日期范围抓取日尺度天气数据  
  Fetch daily weather data by latitude, longitude, and date range
- 支持图形界面与命令行两种使用方式  
  Support both graphical and command-line usage
- 支持轻量级自然语言 weather agent，用于任务解析与自动执行  
  Support a lightweight natural-language weather agent for task parsing and execution
- 支持天气摘要、关键指标洞察和下一步动作建议  
  Support weather summaries, key metric insights, and next-action suggestions
- 支持摄氏度与开尔文两种温度单位  
  Support both Celsius and Kelvin temperature units
- 自动标准化常见天气字段名，便于建模或对比分析  
  Automatically standardize common weather field names for modeling or comparison
- 支持导出抓取到的天气数据 CSV  
  Export fetched weather data as CSV
- 支持上传真实值 CSV 与预测值 CSV 进行误差分析  
  Upload real and predicted CSV files for error analysis
- 自动计算 `MAE`、`RMSE`、`R²`  
  Automatically calculate `MAE`, `RMSE`, and `R²`
- 支持按周、双周、月三个时间尺度查看误差变化  
  View error trends at weekly, biweekly, and monthly scales
- 提供 GitHub Actions 工作流，可自动构建 Windows 可执行文件  
  Includes a GitHub Actions workflow for building a Windows executable

## 使用方式 | Usage Modes

这个项目现在有两个入口模式：  
This project now has two usage modes:

- GUI 模式：启动 `Streamlit` 页面进行交互式操作  
  GUI mode: launch the `Streamlit` interface for interactive use
- CLI 模式：通过命令行参数直接抓取数据并导出 CSV  
  CLI mode: fetch data directly from command-line arguments and export CSV

此外，项目还包含一个轻量级 agent 层，用于把自然语言任务解析成结构化执行计划。  
In addition, the project includes a lightweight agent layer that parses natural-language tasks into structured execution plans.

## Agent Layer

为了让项目更像一个可讲述的智能工具，而不是单纯的数据抓取脚本，仓库中加入了一个简单的 `weather_agent.py`。  
To make the project easier to present as an intelligent tool rather than only a data-retrieval script, the repository now includes a simple `weather_agent.py`.

这个 agent 当前支持：  
The agent currently supports:

- 识别天气抓取任务 / Recognizing weather retrieval tasks
- 识别天气摘要任务 / Recognizing weather summary tasks
- 从自然语言中提取 `latitude`、`longitude`、日期区间和温度单位  
  Extracting `latitude`, `longitude`, date range, and temperature unit from natural language
- 生成结构化任务计划 / Generating a structured task plan
- 输出下一步动作建议 / Producing next-action suggestions
- 生成关键变量摘要和 highlights / Generating variable summaries and highlights
- 在 GUI 中自动执行天气抓取任务 / Executing retrieval tasks automatically in the GUI
- 在 CLI 中通过 `--prompt` 触发天气抓取或摘要 / Triggering retrieval or summary in the CLI through `--prompt`

示例提示词：  
Example prompt:

```text
Summarize the weather for latitude 32 and longitude -84 from 2015-01-01 to 2015-12-31 in Celsius.
```

## 图形界面 | Graphical Interface

图形界面适合交互式查看数据和做真实值/预测值对比分析。  
The graphical interface is suitable for interactive data exploration and real-vs-predicted comparison analysis.

### 界面能力 | What The GUI Does

#### 1. 天气数据获取 | Weather Data Retrieval

用户输入：  
User inputs:

- `Latitude`
- `Longitude`
- `Start Date`
- `End Date`
- 温度单位 / Temperature unit

应用会调用 NASA POWER API 获取日尺度天气数据，并返回标准化后的表格结果。  
The app calls the NASA POWER API to retrieve daily weather data and returns a standardized table.

#### 2. 真实值 vs 预测值对比分析 | Real vs Predicted Comparison

用户可上传两份 CSV：  
Users can upload two CSV files:

- `Real Weather CSV`
- `Predicted Weather CSV`

程序会自动识别两个文件中的公共字段，并允许选择某一列进行对比分析，输出：  
The app automatically identifies shared fields between the two files and lets users choose one field for comparison, then outputs:

- 前 10 行预览 / Preview of the first 10 rows
- 总体 `MAE` / Overall `MAE`
- 总体 `RMSE` / Overall `RMSE`
- 总体 `R²` / Overall `R²`
- 按周误差 / Weekly errors
- 按双周误差 / Biweekly errors
- 按月误差 / Monthly errors
- 绝对误差与原始误差曲线 / Absolute error and raw error curves
- 真实值与预测值折线图 / Actual vs predicted line chart

## 命令行模式 | Command-Line Mode

当你给 `app.py` 传入坐标和日期参数时，它会以命令行模式运行，直接生成天气数据 CSV。  
When you pass coordinates and date arguments to `app.py`, it runs in command-line mode and generates a weather-data CSV directly.

支持参数：  
Supported arguments:

- `--lat`：纬度 / latitude
- `--lon`：经度 / longitude
- `--start`：开始日期，格式 `YYYY-MM-DD` / start date in `YYYY-MM-DD`
- `--end`：结束日期，格式 `YYYY-MM-DD` / end date in `YYYY-MM-DD`
- `--unit`：温度单位，`C` 或 `K` / temperature unit, `C` or `K`
- `--out`：输出 CSV 文件名 / output CSV filename
- `--gui`：显式启动图形界面 / explicitly launch the GUI
- `--prompt`：自然语言任务输入 / natural-language task input

示例：  
Example:

```bash
python app.py --lat 32.0 --lon -84.0 --start 2015-01-01 --end 2015-12-31 --unit C --out weather.csv
```

也可以使用自然语言模式：  
You can also use natural-language mode:

```bash
python app.py --prompt "Summarize the weather for latitude 32 and longitude -84 from 2015-01-01 to 2015-12-31 in Celsius." --out weather.csv
```

运行成功后，程序会输出类似：  
On success, the program prints something like:

```text
Saved: weather.csv  rows=365
```

## 输出字段 | Output Fields

程序会对 NASA POWER 返回列进行标准化，常见输出字段包括：  
The tool standardizes NASA POWER output columns. Common output fields include:

- `date`
- `t_max`
- `t_min`
- `t_avg`
- `t_range`
- `precip`
- `solar_rad`
- `clrsky_solar_rad`
- `toa_solar_rad`
- `rel_humidity`
- `spec_humidity`
- `wind_speed_10m`
- `wind_speed_50m`
- `wind_direction_10m`
- `surface_pressure`
- `unit`

## 项目结构 | Project Structure

```text
weather-tool/
├── app.py                 # Unified entry point for GUI and CLI
├── weather_app.py         # Streamlit UI and interaction logic
├── weather_agent.py       # Lightweight agent plan builder
├── weather_core.py        # Data retrieval, normalization, and evaluation logic
├── requirements.txt       # Python dependencies
├── MeteoAgent.spec        # PyInstaller config
├── .github/workflows/
│   └── windows-build.yml  # Windows EXE build workflow
└── dist/                  # Build artifacts
```

## 安装与运行 | Installation And Run

### 1. 克隆仓库 | Clone the repository

```bash
git clone https://github.com/dchen0212/weather-tool.git
cd weather-tool
```

### 2. 创建虚拟环境 | Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖 | Install dependencies

```bash
pip install -r requirements.txt
```

### 4. 启动图形界面 | Launch the GUI

方式一 / Option 1:

```bash
python app.py
```

方式二 / Option 2:

```bash
python app.py --gui
```

方式三 / Option 3:

```bash
streamlit run weather_app.py
```

默认启动后可在浏览器访问本地 `Streamlit` 页面。  
After startup, the local `Streamlit` page will open in your browser.

### 5. 使用命令行导出 CSV | Use the CLI to export CSV

```bash
python app.py --lat 32.0 --lon -84.0 --start 2015-01-01 --end 2015-12-31 --unit C --out weather.csv
```

### 6. 使用 agent 提示词导出 CSV | Use the agent prompt to export CSV

```bash
python app.py --prompt "Fetch weather for latitude 32 and longitude -84 from 2015-01-01 to 2015-12-31 in Celsius." --out weather.csv
```

## CSV 输入格式说明 | CSV Input Format

用于图形界面对比分析的 CSV 建议满足以下要求：  
For GUI-based comparison analysis, CSV files are recommended to follow these rules:

- 第一行为表头  
  The first row should be the header
- 使用逗号分隔  
  Use comma-separated values
- 数据列为纯数字，不要在单元格中附带单位  
  Numeric values only, with no units inside cells
- 可以包含 `date` 列  
  An optional `date` column is allowed
- `Predicted CSV` 的列名应与真实天气数据输出列名一致，便于自动对齐  
  The `Predicted CSV` should use the same column names as the real weather data output for proper alignment

示例 / Example:

```csv
t_max,t_min,t_avg,precip,solar_rad
298.51,287.43,291.86,0.04,3.5119
299.16,288.33,292.42,3.16,1.1654
295.12,282.39,286.52,15.19,0.9737
```

## 数据来源 | Data Source

本项目当前使用的数据源为：  
This project currently uses:

- [NASA POWER](https://power.larc.nasa.gov/)

调用接口为日尺度点位气象数据接口，适合做天气分析、农业环境分析、建模特征构建等用途。  
The project uses the daily point-based API, which is suitable for weather analysis, agricultural/environmental studies, and feature generation for modeling.

## 打包为可执行文件 | Build As Executable

### 本地使用 PyInstaller 打包 | Build locally with PyInstaller

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name MeteoAgent app.py \
  --collect-all streamlit \
  --copy-metadata streamlit \
  --collect-all altair \
  --copy-metadata altair
```

打包完成后，产物通常位于：  
After packaging, the artifact is typically located at:

```text
dist/MeteoAgent
```

在 Windows 工作流中，产物名称为：  
In the Windows workflow, the output file is:

```text
dist/MeteoAgent.exe
```

## GitHub Actions 自动构建 | GitHub Actions Build

仓库内置了 Windows 构建流程：  
The repository includes a Windows build workflow:

- 文件位置 / File: `.github/workflows/windows-build.yml`
- 触发方式 / Trigger: push to `main` or manual dispatch
- 输出内容 / Outputs:
  - 源码压缩包 / Source archive
  - Windows 可执行文件 `MeteoAgent.exe` / Windows executable `MeteoAgent.exe`

如果你希望向非 Python 用户分发工具，这个流程会比较方便。  
This is useful if you want to distribute the tool to users who do not have a Python environment.

## 依赖说明 | Dependencies

主要依赖包括：  
Main dependencies include:

- `streamlit`
- `pandas`
- `requests`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `chardet`

当前 `requirements.txt` 中还包含 `netCDF4`、`xarray`、`h5netcdf`、`h5py` 等科学计算相关依赖，后续如果不再使用，可按实际需求精简。  
The current `requirements.txt` also includes scientific-computing-related packages such as `netCDF4`, `xarray`, `h5netcdf`, and `h5py`. These can be trimmed later if they are not needed.

## 适用场景 | Use Cases

- 气象数据采集 / Weather data collection
- 天气预测结果评估 / Weather prediction evaluation
- 农业或环境相关数据分析 / Agricultural or environmental data analysis
- 机器学习建模前的数据准备 / Data preparation for machine learning
- 真实值与模型输出的可视化对比 / Visual comparison of ground truth and model outputs
- 需要脚本化导出天气 CSV 的流程 / Workflows that need scripted CSV export
- 简单的任务型 agent 演示项目 / A simple task-oriented agent demo project

## 后续可改进方向 | Possible Improvements

- 增加更多天气数据源作为备用接口 / Add more weather data sources as fallbacks
- 支持批量地点查询 / Support batch location queries
- 增加地图选点功能 / Add map-based location selection
- 增加更多评估指标，如 `MAPE` / Add more evaluation metrics such as `MAPE`
- 支持结果图表一键导出 / Support one-click export of result charts
- 支持时间序列对齐与缺失值处理 / Add time-series alignment and missing-value handling
- 将 CLI 与 GUI 进一步模块化拆分 / Further modularize the CLI and GUI

## License

如果你准备开源发布，建议补充一个明确的许可证文件，例如 `MIT License`。  
If you plan to publish this as an open-source project, it is recommended to add a clear license file, such as the `MIT License`.
