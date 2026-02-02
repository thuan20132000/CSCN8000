# Complete Setup Guide

## Step-by-Step Installation

### 1️⃣ Initialize Git Repository

```bash
cd /Users/thuantruong/Dev/conestoga/CSCN8000
git init
git add .
git commit -m "Initial commit: Setup statistics practice environment"
```

### 2️⃣ Create Python Virtual Environment

**Option A: Using venv (recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify activation (you should see (venv) in prompt)
which python
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n stats-practice python=3.11

# Activate it
conda activate stats-practice
```

### 3️⃣ Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will install:
# - numpy, pandas, scipy (core data science)
# - statsmodels, scikit-learn (statistics & ML)
# - matplotlib, seaborn, plotly (visualization)
# - jupyter, notebook (interactive environment)
```

### 4️⃣ Create Project Directories

```bash
# Create directory structure
mkdir -p notebooks/chapter_{01..10}
mkdir -p data/{raw,processed}
mkdir -p scripts
mkdir -p exercises
mkdir -p projects
mkdir -p resources
```

### 5️⃣ Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab (more features)
pip install jupyterlab
jupyter lab
```

## 🔍 Verify Your Setup

Run the verification notebook:
```bash
jupyter notebook notebooks/getting_started.ipynb
```

Run all cells (Cell → Run All) to verify everything is installed correctly.

## 📂 Recommended Workflow

### Daily Practice
1. **Activate environment**: `source venv/bin/activate`
2. **Start Jupyter**: `jupyter notebook`
3. **Create/open notebook** in appropriate chapter folder
4. **Practice concepts** from the book
5. **Save and commit** progress: `git add . && git commit -m "Completed chapter X exercises"`

### Organizing Your Work
- **notebooks/chapter_XX/**: One notebook per topic/section
- **exercises/**: Solutions to book exercises
- **projects/**: Larger case studies or projects
- **data/**: Store datasets you work with
- **scripts/**: Reusable Python modules/functions

## 🛠️ Useful Commands

### Package Management
```bash
# Add new package
pip install package-name
pip freeze > requirements.txt

# Update packages
pip install --upgrade package-name

# List installed packages
pip list
```

### Git Workflow
```bash
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Your message here"

# View history
git log --oneline

# Create a branch for experiments
git checkout -b experiment-branch
```

### Jupyter Tips
```bash
# Install kernel for your virtual environment
python -m ipykernel install --user --name=stats-practice --display-name="Python (Stats Practice)"

# List available kernels
jupyter kernelspec list

# Convert notebook to Python script
jupyter nbconvert --to script notebook.ipynb
```

## 🐛 Troubleshooting

### Issue: "jupyter: command not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall jupyter
pip install jupyter
```

### Issue: Import errors in notebook
```bash
# Make sure you're using the correct kernel
# In Jupyter: Kernel → Change Kernel → Python (Stats Practice)

# Or reinstall kernel
python -m ipykernel install --user --name=stats-practice --display-name="Python (Stats Practice)" --force
```

### Issue: Matplotlib plots not showing
```python
# Add this at the top of your notebook
%matplotlib inline
```

## 📚 Additional Resources

### Online Documentation
- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Datasets for Practice
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Data.gov](https://data.gov/)
- [FiveThirtyEight Data](https://data.fivethirtyeight.com/)

### Jupyter Extensions (Optional)
```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Useful extensions: Table of Contents, Code Folding, Variable Inspector
```

## 🎯 Next Steps

1. ✅ Complete the setup steps above
2. ✅ Run the getting_started.ipynb notebook
3. 📖 Start with Chapter 1 of your book
4. 💻 Create a notebook: `notebooks/chapter_01/01_introduction.ipynb`
5. 🔄 Commit regularly to track progress

---

**Happy Learning! You're all set to master statistics! 📊**
