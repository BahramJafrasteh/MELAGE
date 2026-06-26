; Inno Setup script for MELAGE
; Build: iscc /DMyAppVersion=2.2.0 packaging\windows\installer.iss

#ifndef MyAppVersion
  #define MyAppVersion "2.2.0"
#endif

[Setup]
AppName=MELAGE
AppVersion={#MyAppVersion}
AppVerName=MELAGE {#MyAppVersion}
AppPublisher=Bahram Jafrasteh
AppPublisherURL=https://github.com/BahramJafrasteh/MELAGE
AppSupportURL=https://github.com/BahramJafrasteh/MELAGE/issues
AppUpdatesURL=https://github.com/BahramJafrasteh/MELAGE/releases
DefaultDirName={autopf}\MELAGE
DefaultGroupName=MELAGE
AllowNoIcons=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=MELAGE-{#MyAppVersion}-windows-setup
SetupIconFile=..\..\assets\resource\main.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\MELAGE\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MELAGE";          Filename: "{app}\MELAGE.exe"; IconFilename: "{app}\MELAGE.exe"
Name: "{group}\Uninstall MELAGE"; Filename: "{uninstallexe}"
Name: "{commondesktop}\MELAGE";  Filename: "{app}\MELAGE.exe"; IconFilename: "{app}\MELAGE.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MELAGE.exe"; Description: "{cm:LaunchProgram,MELAGE}"; Flags: nowait postinstall skipifsilent
