# PlotDSA - EEG Density Spectral Array (DSA) Viewer

PlotDSA is a PySide6 application for real-time EEG visualization, featuring a scrolling Density Spectral Array (DSA) heatmap and live raw EEG trace.

## Quick Start
```powershell
git clone https://github.com/TPVPfaller/PlotDSA.git
cd PlotDSA
```
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m plotdsa.app.main
```


## Key Features
- **Real-time Visualization**: Heatmap (frequency vs. time) and raw EEG trace via the `EEG_DATA` LSL stream.
- **Configurable Settings**:
  - Welch/Multitaper
  - FFT Window Size
  - FFT Window Overlap
  - DSA Color Mapping
  - Maximum Frequency
  - EEG Sweep Speed
- **Data Export/Import**: Saves computed PSD data to CSV under `C:\temp\VSCaptureWave\DSA_Data` (maximum of 24h). Import PSD data with the "Load Data from Time" option.
- **Execution**: Start-DSA-Layout.cmd launches the application with AnesthesiaUI. DSA.exe and VSCaptureWave.exe should
be located in C:\temp\VSCaptureWave. 

## Development
- **Requirements**: Python 3.11, Windows 10 (primary target).
- **Tests**: Run `pytest -q`.
- **Build**: Use Nuitka to produce a Windows executable.

## Build Single-File Executable (Windows)

Install the build dependencies:
```bash
python -m pip install nuitka zstandard
```

Build the executable:
```powershell
python -m nuitka --onefile --standalone `
  --enable-plugin=pyside6 `
  --include-module=PySide6.QtOpenGL `
  --include-data-file=".venv\Lib\site-packages\pylsl\lib\lsl.dll=pylsl/lib/lsl.dll" `
  --windows-icon-from-ico=app_icon.ico `
  --windows-console-mode=disable `
  --windows-product-version=1.0.0.0 `
  --windows-file-version=1.0.0.0 `
  --file-description="EEG Density Spectral Array Viewer" `
  --output-filename=DSA.exe `
  plotdsa\app\main.py
```

Build the emulator as a single-file executable:
```powershell
python -m nuitka --onefile --standalone `
  --include-data-file=".venv\Lib\site-packages\pylsl\lib\lsl.dll=pylsl/lib/lsl.dll" `
  --windows-console-mode=force `
  --output-filename=DSAEmulator.exe `
  plotdsa\tools\emulator.py
```
