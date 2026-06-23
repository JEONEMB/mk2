# CS20 체적 측정기 (volume_scanner)

Synexens CS20 깊이 카메라로 상자의 가로·세로·높이와 체적을 측정하는 Python 프로젝트입니다.

## 처음 실행하는 사람을 위한 안내

### 1. Python 환경 준비

프로젝트 폴더에서 아래 명령을 실행합니다.

```powershell
conda activate mk2
python -m pip install -r requirements.txt
```

Python 3.10 환경을 기준으로 작성되었습니다. `numpy`와 `opencv-python`은 자동으로 설치됩니다.

### 2. Synexens SDK 위치

이 프로젝트는 별도의 `SynexensPythonSDK` 패키지를 사용하지 않습니다. Synexens의 C/C++ SDK에 포함된 `SynexensSDK.dll`을 직접 사용합니다.

가장 쉬운 방법은 SDK 폴더를 아래처럼 두는 것입니다.

```text
C:\SDK\SynexensSDK_4.2.5.0_windows\
├─ SynexensSDK.dll
├─ SonixCamera.dll
├─ csreconstruction2.0.dll
└─ ...
```

`SynexensSDK.dll`이 위 폴더 바로 아래에 있어야 합니다. SDK 폴더 이름은 달라도 괜찮습니다. 다른 위치에 두었다면 실행 전에 다음처럼 경로를 지정합니다.

```powershell
$env:SYNEXENS_SDK_ROOT = "D:\SDK\SynexensSDK_4.2.5.0_windows"
python main.py
```

환경 변수를 매번 입력하기 싫다면 Windows 환경 변수에 `SYNEXENS_SDK_ROOT`를 추가하면 됩니다.

### 3. C# Demo 위치

C# Demo는 카메라 연결을 확인하거나 SDK 예제를 참고하기 위한 자료입니다. Python 체적 측정 실행에는 필수 파일이 아닙니다.

보관을 권장하는 위치는 아래와 같습니다.

```text
volume_scanner/
└─ vendor/
   └─ SynexensCSharpDemo/
      └─ SDKTest/
         └─ bin/x64/Release/
            ├─ SDKTest.exe
            ├─ SynexensSDK.dll
            └─ SonixCamera.dll
```

이 위치에 두면 Python 프로그램도 DLL을 자동으로 찾습니다. 단, SDK/Demo 파일은 용량이 크고 제조사 라이선스가 있을 수 있으므로 Git에 올리지 마세요.

### 4. 실행

CS20을 전원과 USB에 연결한 뒤 실행합니다.

```powershell
python main.py
```

화면에서 사용하는 키는 다음과 같습니다.

- `i`: Depth/IR 화면 전환
- `c`: 빈 플랫폼 보정 및 `data/calibration/platform_plane.json` 저장
- `b`: 상자 후보의 point 수와 ROI만 출력
- `d`: 상자 체적 측정
- `q`: 프로그램 종료

카메라가 없는 상태에서도 화면·측정 파이프라인을 확인하려면 다음 명령을 사용합니다.

```powershell
python main.py --demo
```

## 카메라가 인식되지 않을 때

`CS20 device index 0 was not found; detected 0 device(s)` 메시지는 Python 패키지 문제가 아니라 SDK가 카메라를 찾지 못했다는 뜻입니다.

1. CS20 전원과 USB 케이블을 확인합니다.
2. C# Demo의 `SDKTest.exe`를 실행해 카메라가 인식되는지 먼저 확인합니다.
3. C# Demo도 인식하지 못하면 Synexens 드라이버 또는 Windows 장치 인식 상태를 확인합니다.

## 26/06/23 작업 기록

- SDK 접근을 `camera/`로 한정하고, processing/calibration/measurement를 SDK와 분리했다.
- depth 전처리, point cloud 변환, 플랫폼 평면 보정, depth scale 보정, 상자·윗면·사각형 검출 및 체적 계산 파이프라인을 구현했다.
- C# Demo의 `InitSDK → FindDevice → OpenDevice → StartStreaming → GetLastFrameData → GetIntric` 흐름을 기준으로 `SynexensSDK.dll` ctypes 어댑터를 구현했다.
- CS20은 C# 예제 기준 320x240 depth 및 DEPTHIR 스트림을 사용하도록 설정했다.
- `--demo --headless`로 전체 파이프라인을 검증했다. native DLL 로드는 확인됐으며, 마지막 실장비 점검에서는 CS20이 0대로 탐지되어 장치 연결 또는 드라이버 확인이 필요하다.
