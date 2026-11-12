# pib Robot

Repair and extension of a pib robot with a unified programming interface.

## 🎯 Project Goal

The goal is a humanoid robot that can interact with people through speech, movements, and audiovisual output. A central control system and a Python interface are intended to enable the development of custom applications.

## ⚙️ Planned Features

- **Movement:** Control servos for gestures and other movement sequences precisely.
- **Interaction:** Capture and respond to input via buttons and speech.
- **Audio:** Process and output speech and other sounds using a microphone and speakers, respectively.
- **Display:** Show information and visual content on the screen.
- **Extensions:** Integrate camera functions and optional external AI services in the future.

## 📋 Project Status

- **Repair:** Repairs and basic functional tests of the robot have been carried out.
- **API:** Planning and implementation of the new interface are underway.
- **Evaluation:** Available interfaces and device capabilities are being examined on the robot.

## 🏗️ Planned Architecture

- **Control service:** A central service consolidates access to the hardware.
- **Applications:** A Python library and, in the future, a web interface use the central control system.
- **Hardware integration:** ROS 2 is planned for servos and buttons; audio and camera are to be connected via available robot interfaces.
- **Development:** The control system should be testable without hardware and extensible with additional functionality.

## 🔒 Safety

- **Protective measures:** Movement limits, a central software stop, and securing network access are planned.
- **Hardware tests:** Tests are conducted under supervision. A software stop is no substitute for hardware safety measures.
- **Privacy:** Audio and image data should only be transmitted externally with explicit consent.

## 🔗 Official Information on pib 4 Advanced
- [Download overview for pib 4 Advanced](https://files.pib.rocks/v4/advanced/)



# German Version
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
