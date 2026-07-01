param(
    [switch]$RunSmoke
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $repoRoot
try {
    Write-Host "[deploy] Building React dashboard..."
    Set-Location "UI/components"
    npm run build

    Set-Location $repoRoot
    Write-Host "[deploy] Publishing build artifacts..."
    Copy-Item -Recurse -Force "UI/components/build/*" "build/"

    if ($RunSmoke) {
        Write-Host "[deploy] Running smoke tests..."
        & ".\.venv\Scripts\Activate.ps1"
        python -m pytest tests/test_e2e_integration.py -q
    }

    Write-Host "[deploy] Complete."
} finally {
    Pop-Location
}
