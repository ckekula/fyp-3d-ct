# GGO Project Folder Documentation

This document describes the current project layout so it can be used as a reference when generating or reviewing code.

## Root Layout

```text
ggo-project/
├── backend/
├── frontend/
└── data/
```

## Backend

```text
backend/
├── main.py
├── requirements.txt
├── test.py
└── app/
    ├── routes/
    │   └── process.py
    ├── services/
    │   ├── dicom_loader.py
    │   ├── preprocessing.py
    │   ├── segmentation.py
    │   ├── vessel.py
    │   ├── ggo.py
    │   ├── utils.py
    │   └── test.py
    └── models/
```

### Backend File Roles

- `main.py`: backend application entrypoint.
- `requirements.txt`: Python dependencies for the backend.
- `test.py`: backend launcher for the DICOM test workflow.
- `app/routes/process.py`: route or API handler for processing requests.
- `app/services/dicom_loader.py`: load DICOM slices and volumes.
- `app/services/preprocessing.py`: preprocessing logic for CT data.
- `app/services/segmentation.py`: segmentation routines.
- `app/services/vessel.py`: vessel-related processing.
- `app/services/ggo.py`: ground-glass opacity related processing.
- `app/services/utils.py`: shared helper functions.
- `app/services/test.py`: DICOM reading and volume inspection script.
- `app/models/`: place backend model classes or data schemas here.

## Frontend

```text
frontend/
├── tsconfig.json
└── src/
    ├── App.tsx
    ├── react-app-env.d.ts
    ├── components/
    │   ├── Upload.tsx
    │   ├── Viewer.tsx
    │   └── Controls.tsx
    └── pages/
        └── Home.tsx
```

### Frontend File Roles

- `tsconfig.json`: TypeScript compiler configuration.
- `src/App.tsx`: top-level React app component.
- `src/react-app-env.d.ts`: React type declarations.
- `src/components/Upload.tsx`: file upload UI.
- `src/components/Viewer.tsx`: CT viewing UI.
- `src/components/Controls.tsx`: action buttons and controls.
- `src/pages/Home.tsx`: main landing page.

## Data

```text
data/
└── LIDC-IDRI/
    ├── LICENSE
    ├── LIDC-IDRI-0001/
    ├── LIDC-IDRI-0002/
    ├── LIDC-IDRI-0003/
    ├── LIDC-IDRI-0004/
    ├── LIDC-IDRI-0005/
    ├── LIDC-IDRI-0006/
    ├── LIDC-IDRI-0007/
    └── LIDC-IDRI-0008/
```

### Data Folder Notes

- `data/LIDC-IDRI/`: downloaded LIDC-IDRI dataset content.
- Each `LIDC-IDRI-XXXX/` folder holds a patient study and nested DICOM files.
- `LICENSE`: dataset license file.

## Claude Handoff Prompt

Use this project context when asking Claude to generate code:

```text
I am working in a project called ggo-project with this structure:

- backend: Python backend with routes in app/routes, processing logic in app/services, and entrypoints main.py and test.py.
- frontend: TypeScript React frontend with components in src/components and pages in src/pages.
- data: LIDC-IDRI medical imaging dataset organized by patient folders.

Please generate code that fits this layout and keep backend logic in Python, frontend logic in React + TypeScript, and data loading scripts in backend/app/services or backend as appropriate.
```
