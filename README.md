# 🚑 Legacy AI Healthcare Triage Prototype

> [!IMPORTANT]
> This repository is an archived learning prototype and is no longer actively maintained.
>
> Current development has moved to:
> **[TriageAI — ESI Clinical Intake & Care Routing Assistant](https://github.com/chintan-02/triageai-esi-care-routing)**
>
> This project was created for educational and portfolio demonstration purposes only. It is not clinically validated, does not provide medical diagnosis, and must not be used for real patient-care decisions.

<p align="center">
  <b>Early End-to-End Machine Learning and Flask Application</b><br>
  Demonstrating RED, YELLOW, and GREEN priority classification using a Logistic Regression model and rule-based escalation.
</p>

---

## Project Status

- **Status:** Archived legacy prototype
- **Original purpose:** Learning full-stack ML application development
- **Current replacement:** [TriageAI ESI Care Routing](https://github.com/chintan-02/triageai-esi-care-routing)
- **Clinical use:** Not permitted
- **Maintenance:** No longer actively maintained

This repository represents an earlier stage of my development before moving to a more structured React, FastAPI, LightGBM, database-backed, human-in-the-loop clinical decision-support workflow.

---

## 🌐 Legacy Demonstration

The original prototype was deployed using Azure App Service:

**Legacy URL:**  
https://ai-triage-chintan.azurewebsites.net

> The deployment may be unavailable and does not represent the architecture, model, safety workflow, or interface of the current TriageAI project.

---

## Project Overview

This prototype explores how a basic machine-learning model can be integrated into a web application for educational healthcare workflow demonstrations.

The system:

- Accepts a small set of patient intake fields
- Uses Logistic Regression to generate a RED, YELLOW, or GREEN priority classification
- Applies rule-based escalation for selected high-risk inputs
- Displays a confidence score and decision-support result
- Stores demo patient records
- Provides basic analytics and CSV export functionality
- Uses a Flask-based user interface

The RED, YELLOW, and GREEN labels are project-specific demonstration categories. They are not a validated implementation of CTAS, ESI, or another official clinical triage standard.

---

## Key Features

- Patient priority classification using Logistic Regression
- Rule-based escalation for selected vital-sign thresholds
- Flask-based prediction workflow
- Prediction confidence display
- Demo patient history
- Basic analytics dashboard
- CSV export
- Demonstration login workflow
- Azure App Service deployment experience

---

## Machine-Learning Approach

### Model

The prototype uses a Logistic Regression model built with scikit-learn.

Input features include:

- Age
- Pain level
- Respiratory rate
- Heart rate
- Oxygen saturation

### Rule-Based Escalation

The application contains simple escalation rules for selected inputs, including:

- Oxygen saturation at or below a configured threshold
- High reported pain level
- Elevated respiratory rate
- Other configured critical-vital conditions

These rules were created for workflow demonstration only.

> Rule-based escalation does not guarantee clinical safety, prevent all misclassification, or replace professional assessment.

---

## High-Level Architecture

```text
Patient Input
    ↓
Input Validation
    ↓
Rule-Based Escalation
    ↓
Logistic Regression Model
    ↓
Priority Classification
    ↓
Flask Dashboard
```

---

## Screenshots

### Login Page

![Login](screenshots/login.png)

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Patient List

![Patients](screenshots/patient_list.png)

### Analytics

![Analytics](screenshots/analytics.png)

### RED Demonstration Case

![RED](screenshots/patient_red.png)

### YELLOW Demonstration Case

![YELLOW](screenshots/patient_yellow.png)

### GREEN Demonstration Case

![GREEN](screenshots/patient_green.png)

---

## Project Structure

```text
prioritycare/
│
├── app.py              # Flask application and dashboard
├── auth.py             # Demonstration authentication routes
├── patients.py         # Patient CRUD and prediction workflow
├── analytics.py        # Analytics API and CSV export
├── models.py           # Database models
├── config.py           # Application configuration
├── seed.py             # Demo database seeding
├── requirements.txt
├── Procfile            # Azure/Gunicorn deployment configuration
│
├── ml/
│   ├── predict.py      # Model inference and escalation logic
│   └── triage_model.pkl
│
├── templates/          # Jinja HTML templates
├── static/             # CSS and JavaScript
├── notebooks/          # Model-development notebook
└── screenshots/        # Application screenshots
```
