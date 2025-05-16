# VSCode + Scikit-learn CardQuick reference

## 🔧 VSCode Setup

1. Install Python Extension

    - Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
    - Install Python by Microsoft

2. Enable IntelliSense with Pylance
    -Open Command Palette: Ctrl+Shift+P / Cmd+Shift+P
    - Choose: Preferences: Open Settings (JSON)
    - Add:
    `"python.languageServer": "Pylance"`

## 💡 Productivity Tricks

- ☞ Hover Tooltips
  - Hover over any class/function to view documentation and parameter descriptions
- ☞ Signature Help
  - Type ( after any method (like train_test_split() to see full signature
  - Press Ctrl+Space to manually trigger help
- ☞ Go to Definition / Source
  - Ctrl+Click / Cmd+Click on any method/class to jump to its source
    - Or use F12 while your cursor is on a method name
- ☞ View Docstring Inline
  - Hover over LogisticRegression, Pipeline, etc. to read full sklearn docstrings without leaving the notebook

## 📚 Common Imports with Purpose

```python
import pandas as pd                    # Tables & DataFrames
import numpy as np                     # Vectors, math ops
import matplotlib.pyplot as plt        # Graphs

from sklearn.model_selection import train_test_split  # Train/test data
from sklearn.pipeline import Pipeline                  # ML workflow
from sklearn.preprocessing import StandardScaler       # Scaling
from sklearn.linear_model import LinearRegression      # Models
from sklearn.metrics import mean_absolute_error        # Evaluation
```

📊 Quick Sklearn Pipeline Example

```python
def pipeline_linear():
    return Pipeline([
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])

pipeline = pipeline_linear()
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
```

## ✨ Tips

- Use df.info() and df.describe() to explore data fast
- Assign your pipeline to a variable like pipe or model so you can reuse
- Use .score(), .predict(), .fit() consistently

## 🌐 Extra Tools

- Python Docstring Generator (VSCode extension)
- Jupyter extension for notebooks
- Use Notion/Markdown notes beside your code to cheat without memorizing

## Spencer's Teaching Tip

You don’t need to memorise methods. Just learn the shape of the problem. VSCode can take care of the rest.
