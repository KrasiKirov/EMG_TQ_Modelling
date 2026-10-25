param([string]$Slides = "")
$pptx = "C:\Users\dramo\Desktop\Projects\krasi_ml\EMG_TQ_Modelling\.claude\worktrees\dorsiflexion-r2-drop-341137\notes\final_pipeline\presentation\EMG_torque_supervisor_v2.pptx"
$out = "C:\Users\dramo\AppData\Local\Temp\claude\C--Users-dramo-Desktop-Projects-krasi-ml-EMG-TQ-Modelling--claude-worktrees-dorsiflexion-r2-drop-341137\1399efbf-302c-4db2-a720-49b2b753a65e\scratchpad\deck\v2\render"
New-Item -ItemType Directory -Force $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($pptx, $true, $false, $false)
$n = $p.Slides.Count
if ($Slides -eq "") { $list = 1..$n } else { $list = $Slides.Split(',') | ForEach-Object { [int]$_ } }
foreach ($i in $list) { $p.Slides.Item($i).Export((Join-Path $out ("s{0:D2}.png" -f $i)), 'PNG', 1600, 900) }
$p.Close(); $app.Quit()
"exported $($list.Count) of $n"
