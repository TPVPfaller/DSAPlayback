# PlotDSA - EEG Density Spectral Array (DSA) Viewer

DSAPlayback is the playback version of PlotDSA: a PySide6 application that visualizes recorded EEG data from CSV files, featuring a Density Spectral Array (DSA) heatmap and raw EEG trace.

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
- **EEG File Playback**: Open an EEG recording (CSV) via "Load EEG File (CSV)" or by dropping it onto the DSA.
  - **Load**: computes the whole DSA at once.
  - **Play**: replays the recording in real time (heatmap and raw EEG trace).
- **Configurable Settings**:
  - Welch/Multitaper
  - FFT Window Size
  - FFT Window Overlap
  - DSA Color Mapping
  - Maximum Frequency
  - EEG Sweep Speed
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
  --windows-icon-from-ico=app_icon.ico `
  --windows-console-mode=disable `
  --windows-product-version=1.0.0.0 `
  --windows-file-version=1.0.0.0 `
  --file-description="EEG Density Spectral Array Viewer" `
  --output-filename=DSA.exe `
  plotdsa\app\main.py
```
