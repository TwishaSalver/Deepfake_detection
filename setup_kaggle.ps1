# Kaggle API Key Setup Script
# Run this after you get your API key

param(
    [string]$ApiKey = "",
    [string]$ApiUsername = ""
)

if ([string]::IsNullOrEmpty($ApiKey) -or [string]::IsNullOrEmpty($ApiUsername)) {
    Write-Host "Usage: .\setup_kaggle.ps1 -ApiUsername 'your_username' -ApiKey 'your_api_key'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps to get your Kaggle API key:" -ForegroundColor Cyan
    Write-Host "1. Go to https://www.kaggle.com/settings/account" -ForegroundColor Green
    Write-Host "2. Click 'Create New API Token'" -ForegroundColor Green
    Write-Host "3. Download kaggle.json" -ForegroundColor Green
    Write-Host "4. Extract username and key from the file" -ForegroundColor Green
    Write-Host ""
    Write-Host "Then run:" -ForegroundColor Yellow
    Write-Host ".\setup_kaggle.ps1 -ApiUsername 'your_username' -ApiKey 'your_api_key'" -ForegroundColor Yellow
    exit
}

# Create .kaggle directory
$KaggleDir = "$env:USERPROFILE\.kaggle"
if (-not (Test-Path $KaggleDir)) {
    New-Item -ItemType Directory -Path $KaggleDir | Out-Null
    Write-Host "Created $KaggleDir" -ForegroundColor Green
}

# Create kaggle.json
$KaggleJson = @{
    username = $ApiUsername
    key = $ApiKey
} | ConvertTo-Json

$JsonPath = "$KaggleDir\kaggle.json"
$KaggleJson | Out-File -FilePath $JsonPath -Encoding UTF8
Write-Host "Created $JsonPath" -ForegroundColor Green

# Set permissions (Windows)
icacls $JsonPath /inheritance:r | Out-Null
icacls $JsonPath /grant:r "${env:USERNAME}:F" | Out-Null
Write-Host "Set file permissions" -ForegroundColor Green

Write-Host ""
Write-Host "✅ Kaggle API setup complete!" -ForegroundColor Green
Write-Host "Ready to train with real data from Kaggle" -ForegroundColor Cyan
