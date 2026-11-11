# pib-Roboter

Instandsetzung und Erweiterung eines pib-Roboters mit einer einheitlichen Programmierschnittstelle.

## 🎯 Projektziel

Ziel ist ein menschenähnlicher Roboter, der über Sprache, Bewegungen und audiovisuelle Ausgaben mit Menschen interagieren kann. Eine zentrale Steuerung und eine Python-Schnittstelle sollen die Entwicklung eigener Anwendungen ermöglichen.

## ⚙️ Geplanter Funktionsumfang

- **Bewegung:** Servos für Gesten und andere Bewegungsabläufe gezielt ansteuern.
- **Interaktion:** Eingaben über Knöpfe und Sprache erfassen und darauf reagieren.
- **Audio:** Sprache und andere Töne über Mikrofon und Lautsprecher verarbeiten beziehungsweise ausgeben.
- **Anzeige:** Informationen und visuelle Inhalte auf dem Bildschirm darstellen.
- **Erweiterungen:** Perspektivisch Kamerafunktionen und optionale externe KI-Dienste einbinden.

## 📋 Projektstand

- **Instandsetzung:** Reparatur und grundlegende Funktionstests des Roboters wurden durchgeführt.
- **API:** Planung und Implementierung der neuen Schnittstelle laufen.
- **Prüfung:** Verfügbare Schnittstellen und Gerätefähigkeiten werden am Roboter untersucht.

## 🏗️ Geplante Architektur

- **Steuerungsdienst:** Ein zentraler Dienst bündelt den Zugriff auf die Hardware.
- **Anwendungen:** Eine Python-Bibliothek und perspektivisch eine Web-Oberfläche nutzen die zentrale Steuerung.
- **Hardwareanbindung:** ROS 2 ist für Servos und Knöpfe vorgesehen; Audio und Kamera sollen über verfügbare Roboterschnittstellen angebunden werden.
- **Entwicklung:** Die Steuerung soll ohne Hardware testbar und um zusätzliche Funktionen erweiterbar sein.

## 🔒 Sicherheit

- **Schutzmaßnahmen:** Bewegungsgrenzen, ein zentraler Software-Stopp und die Absicherung der Netzwerkzugänge sind geplant.
- **Hardwaretests:** Tests erfolgen unter Aufsicht. Ein Software-Stopp ersetzt keine hardwareseitigen Schutzmaßnahmen.
- **Datenschutz:** Audio- und Bilddaten sollen nur mit ausdrücklicher Freigabe extern übertragen werden.

## 🔗 Offizielle Informationen zu pib 4 Advanced
- [Download-Übersicht für pib 4 Advanced](https://files.pib.rocks/v4/advanced/)
