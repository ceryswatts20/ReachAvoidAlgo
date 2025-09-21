## Setup

- Have Python 3.13

### 1. Open a terminal

- On Windows, use PowerShell (run as administrator).
- On Linux/macOS, use your regular shell.

### 2. Navigate to project folder

Example

```powershell
cd "C:\Users\<YourName>\path\to\project\Robotic_Manipulator_Analysis"
```

### 3. Create a virtual enviroment and install required packages

- This creates a folder called `.venv` inside your project, activates it and installs the required packages.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
