$ErrorActionPreference = 'Stop'

Write-Host "[quality] running tests..."
d:/AI_Model_Performance_Metrics/.venv/Scripts/python.exe -m pytest -q

Write-Host "[quality] running pipeline smoke..."
d:/AI_Model_Performance_Metrics/.venv/Scripts/python.exe -m src.cli --config config/config.yaml

Write-Host "[quality] done"
