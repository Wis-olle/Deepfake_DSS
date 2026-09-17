# Chạy ba detector nối tiếp, tách khỏi phiên làm việc (không bị giới hạn thời gian của công cụ).
# Dùng:  powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$dfb = 'E:\AI Agents\neural\deepfake\dfb'
$out = 'E:\AI Agents\neural\deepfake\out\dfb'
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($d in 'small_cnn', 'resnet18', 'sbi') {
  "===== $d $(Get-Date -Format HH:mm:ss) =====" | Add-Content "$out\run_all.log"
  $p = Start-Process -FilePath 'E:\AI Agents\neural\venv\Scripts\python.exe' `
        -ArgumentList @('train.py', '--detector_path', "config/detector/$d.yaml") -WorkingDirectory $dfb `
        -RedirectStandardOutput "$out\$d.stdout.log" -RedirectStandardError "$out\$d.stderr.log" -NoNewWindow -PassThru -Wait
  "exit $($p.ExitCode) $(Get-Date -Format HH:mm:ss)" | Add-Content "$out\run_all.log"
}
"DONE $(Get-Date -Format HH:mm:ss)" | Add-Content "$out\run_all.log"
