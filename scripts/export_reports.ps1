param([string[]]$Folders=@('lab01_representations','lab02_annotation','lab03_classification','lab04_prompt_program','lab05_mini_rag','lab06_ai_detective'))
$ErrorActionPreference='Stop'
function Export-Report([string]$folder,[string]$stem) {
 $wordApp=New-Object -ComObject Word.Application
 $reportDoc=$null
 try {
  $wordApp.Visible=$false
  $wordApp.DisplayAlerts=0
  $wordApp.AutomationSecurity=3
  $docPath=[string](Join-Path (Get-Location) ($folder+'/'+$stem+'.docx'))
  $pdfPath=[string](Join-Path (Get-Location) ($folder+'/'+$stem+'.pdf'))
  $reportDoc=$wordApp.Documents.OpenNoRepairDialog($docPath,$false,$false)
  Write-Output ('Opened '+$docPath)
  if($reportDoc.TablesOfContents.Count -gt 0){$reportDoc.TablesOfContents.Item(1).Update()}
  $reportDoc.Save()
  $reportDoc.SaveAs2($pdfPath,17)
  Write-Output ('Saved '+$pdfPath)
 } finally {
  if($null -ne $reportDoc){$reportDoc.Close(0);[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportDoc)}
  $wordApp.Quit()
  [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp)
 }
}
foreach($folder in $Folders) {
 Export-Report $folder 'report'
 if($folder -eq 'lab02_annotation'){Export-Report $folder 'guidelines'}
}
