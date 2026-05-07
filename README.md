[README.md](https://github.com/user-attachments/files/27474469/README.md)
# Weather Tool

一个基于 `Streamlit` 的天气数据获取与分析工具，支持从 **NASA POWER** 拉取指定经纬度和时间范围的日尺度气象数据，并对真实值与预测值 CSV 进行误差评估与可视化分析。  
A `Streamlit`-based weather data acquisition and analysis tool. It supports fetching daily weather data for a given latitude/longitude and date range from **NASA POWER**, and provides error analysis and visualization for real vs predicted CSV files.

仓库地址 / Repository: [dchen0212/weather-tool](https://github.com/dchen0212/weather-tool)

## 功能简介 | Features

- 根据经纬度和日期范围获取天气数据  
  Fetch weather data by latitude, longitude, and date range
- 支持摄氏度与开尔文两种温度单位  
  Support both Celsius and Kelvin temperature units
- 自动标准化常见天气字段名，便于建模或对比分析  
  Automatically standardize common weather field names for modeling or comparison
- 支持上传真实值 CSV 与预测值 CSV 进行误差分析  
  Upload real and predicted CSV files for error analysis
- 自动计算 `MAE`、`RMSE`、`R²`  
  Automatically calculate `MAE`, `RMSE`, and `R²`
- 支持按周、双周、月三个时间尺度查看误差变化  
  View error trends at weekly, biweekly, and monthly scales
- 生成真实值/预测值折线图、误差曲线图和分段指标图  
  Generate actual-vs-predicted charts, error curves, and interval metric plots
- 支持导出抓取到的天气数据 CSV  
  Export fetched weather data as CSV
- 提供 GitHub Actions 工作流，可自动构建 Windows 可执行文件  
  Includes a GitHub Actions workflow for building a Windows executable

## 界面能力 | What The App Does

应用主要分为两部分。  
The application is mainly divided into two parts.

### 1. 天气数据获取 | Weather Data Retrieval

用户输入：  
User inputs:

- `Latitude`
- `Longitude`
- `Start Date`
- `End Date`
- 温度单位 / Temperature unit

应用会调用 NASA POWER API 获取日尺度天气数据，并返回标准化后的表格结果。  
The app calls the NASA POWER API to retrieve daily weather data and returns a standardized table.

当前主要输出字段包括：  
Current main output fields include:

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

### 2. 真实值 vs 预测值对比分析 | Real vs Predicted Comparison

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

## 项目结构 | Project Structure

```text
weather-tool/
├── app.py                 # Streamlit launcher
├── weather_app.py         # UI and interaction logic
├── weather_core.py        # Data retrieval, normalization, and evaluation logic
├── requirements.txt       # Python dependencies
├── WeatherTool.spec       # PyInstaller config
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

### 4. 启动应用 | Start the app

方式一 / Option 1:

```bash
streamlit run weather_app.py
```

方式二 / Option 2:

```bash
python app.py
```

默认启动后可在浏览器访问本地 `Streamlit` 页面。  
After startup, the local `Streamlit` page will open in your browser.

## CSV 输入格式说明 | CSV Input Format

用于对比分析的 CSV 建议满足以下要求：  
For comparison analysis, CSV files are recommended to follow these rules:

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
pyinstaller --onefile --noconsole --name WeatherTool app.py \
  --collect-all streamlit \
  --copy-metadata streamlit \
  --collect-all altair \
  --copy-metadata altair
```

打包完成后，产物通常位于：  
After packaging, the artifact is typically located at:

```text
dist/WeatherTool
```

在 Windows 工作流中，产物名称为：  
In the Windows workflow, the output file is:

```text
dist/WeatherTool.exe
```

## GitHub Actions 自动构建 | GitHub Actions Build

仓库内置了 Windows 构建流程：  
The repository includes a Windows build workflow:

- 文件位置 / File: `.github/workflows/windows-build.yml`
- 触发方式 / Trigger: push to `main` or manual dispatch
- 输出内容 / Outputs:
  - 源码压缩包 / Source archive
  - Windows 可执行文件 `WeatherTool.exe` / Windows executable `WeatherTool.exe`

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

## 后续可改进方向 | Possible Improvements

- 增加更多天气数据源作为备用接口 / Add more weather data sources as fallbacks
- 支持批量地点查询 / Support batch location queries
- 增加地图选点功能 / Add map-based location selection
- 增加更多评估指标，如 `MAPE` / Add more evaluation metrics such as `MAPE`
- 支持结果图表一键导出 / Support one-click export of result charts
- 支持时间序列对齐与缺失值处理 / Add time-series alignment and missing-value handling

## License

如果你准备开源发布，建议补充一个明确的许可证文件，例如 `MIT License`。  
If you plan to publish this as an open-source project, it is recommended to add a clear license file, such as the `MIT License`.
