# Start a QD exploration in the background on this PC (hidden WSL window; survives closing terminals).
#   powershell -File tools/qd_background.ps1 specs/qd/wide-net-v1.qd.json [-Hours 20]
# Watch / control (any shell):
#   python tools/qd_explore.py status sweep_runs/QD_WIDE_NET_V1     (or open status.json / log.txt)
#   python tools/qd_explore.py pause|resume|stop sweep_runs/QD_WIDE_NET_V1
# A stopped run continues where it left off when started again with the same config.
param([Parameter(Mandatory = $true)][string]$Config, [double]$Hours = 0)
$repo = Split-Path -Parent $PSScriptRoot
$drive = $repo.Substring(0, 1).ToLower()
$wslRepo = "/mnt/$drive" + $repo.Substring(2).Replace('\', '/')
$wslArgs = @("bash", "$wslRepo/tools/qd_run_wsl.sh", $Config.Replace('\', '/'))
if ($Hours -gt 0) { $wslArgs += @("--hours", "$Hours") }
Start-Process -FilePath "wsl.exe" -ArgumentList $wslArgs -WindowStyle Hidden -WorkingDirectory $repo
Write-Output "started: $Config (log: sweep_runs/QD_<ID>/log.txt)"
