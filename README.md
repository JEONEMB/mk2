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

CS20은 플랫폼 위 약 70 cm 높이에 설치하고, 카메라가 아래 플랫폼을 향하도록 고정한 뒤 실행합니다. 정면으로 벽과 책상면을 동시에 보는 설치는 이 체적 측정 모드의 보정 대상이 아닙니다.

```powershell
python main.py
```

처음에는 Depth 창만 표시됩니다. `i` 키를 누르면 독립된 IR 창이 추가로 열리고, 다시 `i`를 누르면 닫힙니다. 각 창 왼쪽 위에서 실제 해상도를 확인할 수 있습니다.

화면에서 사용하는 키는 다음과 같습니다.

- `i`: IR 창 열기/닫기
- `r`: 320×240 ↔ 640×480 해상도 전환. 전환 뒤에는 `c`로 다시 보정
- `h`: 보정 후 플랫폼 대비 상대높이 진단 화면 열기/닫기. 0 mm는 차가운 색, +70 mm는 따뜻한 색
- `p`: 빈 플랫폼 60-frame 평면 proposal 미리보기. 초록색은 반복 검증된 평면 inlier이고, 노란색 ROI는 모든 품질 조건을 통과한 경우에만 표시
- `c`: `p`와 같은 60-frame proposal을 확인한 뒤 `Enter`/`Space`로 확정하면 줄자로 잰 렌즈 중심-플랫폼 높이(`700` 또는 `70cm`)를 입력해 보정을 저장. `c`/`Esc`는 취소
- `b`: 모든 상자 후보의 높이·point 수를 출력하고 후보 mask를 화면에 표시
- `d`: 상자 체적 측정
- `q`: 프로그램 종료

프로그램을 시작하면 ROI는 보이지 않습니다. 빈 플랫폼에서 먼저 `p`를 눌러 초록색 consensus plane mask와 노란색 제안 ROI를 확인합니다. 노란색 ROI가 표시되지 않으면 저장 가능한 플랫폼 영역이 없다는 뜻이므로 보정을 진행하지 않습니다. 올바른 ROI가 표시되면 `c`를 누르고 `Enter`/`Space`로 같은 proposal을 확정한 뒤 정확한 높이(`700` 또는 `70cm`)를 입력합니다. ROI에는 16 px 안전 여백이 적용됩니다. `d`는 현재 프레임의 플랫폼 기준면을 다시 맞춘 뒤, 그 기준면보다 40 mm 이상 위인 ROI 내부 점만 상자 후보로 사용합니다.

`c`가 끝나면 콘솔은 `SDK plane`, `measured`, `scale`, `corrected` 높이를 모두 출력하고 화면의 Platform ROI에도 보정된 높이를 표시합니다. `SDK plane`은 카메라 원점에서 평면까지의 SDK 거리이고, `corrected`는 줄자로 잰 렌즈 중심 높이를 기준으로 보정된 값입니다. 플랫폼 mask는 측정 내부에서만 사용하며, 화면에는 노란색 사각형 ROI만 표시합니다.

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

## 2026/06/24 작업 기록

- 상자 측정의 윗면 최소 점 수를 `300`으로 높였다. `b`(검출) 또는 `d`(측정) 후 화면에서 주황색은 모든 높이 후보 mask, 초록색은 선택된 후보이며, 각 후보에 `C번호 H높이(mm) P점수`가 표시된다. 주황색/빨간 테두리 후보는 점 수 부족 또는 플랫폼 경계 접촉으로 제외된 후보이다.

- 실장비 빈 플랫폼 보정 결과 `camera_height_mm=526.42`가 기록됐다. 설치 높이를 700 mm로 가정하면 `173.58 mm` 낮고, 오차율은 약 `24.80%`다.
- 기존 값은 평면식 `n·x + d = 0`에서 `abs(d)`로 계산한 카메라 원점-평면 최단 거리였다. 당시에는 `GetLastFrameData`의 raw 16-bit depth를 곧바로 mm로 간주해 점군을 만들었으므로 절대 거리 scale이 맞지 않을 수 있었다.
- SDK의 `GetDepthPointCloud` metric XYZ를 사용하도록 변경했다. `c` 시 실제 렌즈 중심-빈 플랫폼 거리(mm)를 입력하면 관측값 대비 scale을 저장하고, scale 적용 점군으로 플랫폼 평면을 다시 보정한다.
- 플랫폼의 largest plane component mask를 `platform_plane_mask.npy`에 함께 저장한다. 상자 후보는 이 mask 안에서만 찾으며 mask 경계에 닿으면 측정을 실패 처리해 배경·벽·ROI 밖 성분의 오검출을 막는다.
- 정확도 확인 순서: 빈 플랫폼 → `c` → 정확한 실측 높이 입력 → 최종 camera height와 mask 확인 → 상자 배치 → `d`.
- `70`을 입력하면 이전에는 70 mm로 처리되어 scale `0.12736`, 최종 높이 약 68.98 mm가 저장됐다. 이제 `70`은 70 cm(700 mm)로 자동 해석하며, `70cm`처럼 단위를 명시할 수도 있다.
- SDK의 `GetDepthPointCloud`와 `GetIntric`를 모두 `bUndistort=True` 좌표계로 전환했다. raw point cloud는 preview에서 왜곡 보정 전후 평면 잔차를 비교하는 진단용으로만 유지한다.
- `p`/`c`는 같은 빈 플랫폼 60-frame 묶음에서 수평 평면 후보를 만들고, 프레임 간 반복 inlier만 consensus mask로 사용한다. 사용자가 보는 proposal과 저장되는 ROI는 동일한 frame 묶음에서 나온다.
- 후보 component는 raw 8-connected inlier mask로만 구분한다. 기존 41×41 closing으로 떨어진 섬을 합치던 처리를 제거했고, 단일 component 내부의 5 px 이하 hole만 ROI 탐색 시 제한적으로 허용한다.
- 노란색 ROI는 단일 component 안에서 90% 이상 plane coverage, 최소 크기·안전 여백, 최종 평면 잔차를 모두 통과할 때만 표시·저장한다. 조건을 만족하지 못하면 ROI를 그리지 않고 component/coverage/잔차 단계의 실패 사유를 출력한다.
- `tests/`에 undistorted SDK 좌표 요청, 전경을 피하는 ROI, 가까운 평면 섬 미병합을 검증하는 단위 테스트를 추가했다.
- raw 16-bit IR에 250 고정값을 적용하면 밝은 플랫폼 depth 대부분이 제거될 수 있어, frame별 상위 0.2%만 glare로 제거하는 적응형 IR saturation mask로 변경했다. dynamic range가 작거나 mask가 화면의 5%를 넘으면 IR mask를 적용하지 않는다.
- 오버헤드 보정 시 valid depth 비율, platform normal 정렬, ROI 면적을 검증한다. 유효 depth가 화면의 15% 미만이거나 ROI가 화면의 10% 미만이면 calibration 파일을 저장하지 않고 카메라·IR·설치 방향을 점검하도록 안내한다.
