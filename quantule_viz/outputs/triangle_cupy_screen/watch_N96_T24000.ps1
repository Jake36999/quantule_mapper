$csv = 'F:\quantule_mapper\quantule_viz\outputs\triangle_cupy_screen\triangle_spacing_screen_N96_T24000_results.csv'
$report = 'F:\quantule_mapper\quantule_viz\outputs\triangle_cupy_screen\triangle_spacing_screen_N96_T24000_report.md'
$manifest = 'F:\quantule_mapper\quantule_viz\outputs\triangle_cupy_screen\screen_manifest_N96_T24000.json'
$marker = 'F:\quantule_mapper\quantule_viz\outputs\triangle_cupy_screen\watch_N96_T24000_done.txt'
$errorMarker = 'F:\quantule_mapper\quantule_viz\outputs\triangle_cupy_screen\watch_N96_T24000_error.txt'
while ($true) {
  if ((Test-Path $csv) -and (Test-Path $report) -and (Test-Path $manifest)) {
    $rows = Import-Csv $csv
    if ($rows.Count -ge 2) {
      "done $(Get-Date -Format o) rows=$($rows.Count)" | Set-Content -Path $marker
      break
    }
  }
  $tri = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*triangle_cupy_screen.py*--N 96*--steps 24000*' }
  if (-not $tri -and -not (Test-Path $csv)) {
    "process_missing_without_csv $(Get-Date -Format o)" | Set-Content -Path $errorMarker
    break
  }
  Start-Sleep -Seconds 30
}
