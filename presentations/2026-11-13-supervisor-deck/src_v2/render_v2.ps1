param([string]$Slides = "")
$pptx = Join-Path $PSScriptRoot "..\EMG_torque_supervisor_v2.pptx"
$out = Join-Path $env:TEMP "deck_v2_render"
New-Item -ItemType Directory -Force $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($pptx, $true, $false, $false)
$n = $p.Slides.Count
if ($Slides -eq "") { $list = 1..$n } else { $list = $Slides.Split(',') | ForEach-Object { [int]$_ } }
foreach ($i in $list) { $p.Slides.Item($i).Export((Join-Path $out ("s{0:D2}.png" -f $i)), 'PNG', 1600, 900) }
$p.Close(); $app.Quit()
"exported $($list.Count) of $n"
