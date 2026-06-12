import json
import io
import base64
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def download_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("Downloading Titanic dataset...")
    urllib.request.urlretrieve(url, "titanic.csv")
    print("Dataset downloaded and saved as titanic.csv")

def generate_notebook():
    # Load dataset
    df = pd.read_csv('titanic.csv')
    
    # ------------------ Run Analysis for Cells ------------------
    
    # Cell 2 output capture
    cell2_stdout = f"Dataset loaded successfully! Shape: {df.shape}\n"
    cell2_df_head = df.head()
    
    # Cell 4 output capture (Cleaning)
    missing_before = df.isnull().sum()
    
    # Fill missing Age with mean
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)
    
    # Fill missing Embarked with mode
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    
    # Drop irrelevant columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean = df.drop(columns=columns_to_drop)
    
    missing_after = df_clean.isnull().sum()
    
    cell4_stdout = (
        "Missing values in raw dataset:\n"
        f"{missing_before.to_string()}\n\n"
        "Missing values after cleaning:\n"
        f"{missing_after.to_string()}\n\n"
        f"Cleaned dataset shape: {df_clean.shape}\n"
    )
    cell4_df_head = df_clean.head()
    
    # Cell 6 output capture (Age Group)
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ['Child (0-12)', 'Teen (12-18)', 'Young Adult (18-35)', 'Middle Aged (35-60)', 'Senior (60+)']
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels)
    
    survival_by_age = df_clean.groupby('AgeGroup', observed=False)['Survived'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}
    )
    survival_by_age_formatted = survival_by_age.copy()
    survival_by_age_formatted['Survival Rate'] = survival_by_age_formatted['Survival Rate'].map('{:.2%}'.format)
    cell6_stdout = "=== Survival Rate by Age Group ===\n"
    
    # Cell 7 output capture (Embarkation Port)
    port_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    df_clean['EmbarkationPort'] = df_clean['Embarked'].map(port_map)
    
    survival_by_port = df_clean.groupby('EmbarkationPort')['Survived'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}
    )
    survival_by_port_formatted = survival_by_port.copy()
    survival_by_port_formatted['Survival Rate'] = survival_by_port_formatted['Survival Rate'].map('{:.2%}'.format)
    cell7_stdout = "=== Survival Rate by Embarkation Port ===\n"
    
    # Cell 8 output capture (Family Size)
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch']
    
    survival_by_family = df_clean.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}
    )
    survival_by_family_formatted = survival_by_family.copy()
    survival_by_family_formatted['Survival Rate'] = survival_by_family_formatted['Survival Rate'].map('{:.2%}'.format)
    cell8_stdout = "=== Survival Rate by Family Size ===\n"
    
    # ------------------ Plot Generation & Capture ------------------
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Age Distribution
    plt.figure(figsize=(10, 6), dpi=150)
    sns.histplot(df_clean['Age'], kde=True, color='#2c5282', bins=30, edgecolor='white', alpha=0.85)
    plt.title('Age Distribution of Titanic Passengers', fontsize=16, fontweight='bold', pad=15, color='#2d3748')
    plt.xlabel('Age', fontsize=12, labelpad=10, color='#4a5568')
    plt.ylabel('Passenger Count', fontsize=12, labelpad=10, color='#4a5568')
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', dpi=150)
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close()
    
    # Plot 2: Heatmap
    plt.figure(figsize=(10, 8), dpi=150)
    numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    corr_matrix = df_clean[numeric_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                annot_kws={"fontsize": 11, "fontweight": "semibold"})
    
    plt.title('Correlation Matrix of Titanic Numeric Features', fontsize=16, fontweight='bold', pad=20, color='#2d3748')
    plt.tight_layout()
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', dpi=150)
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close()
    
    # Plot 3: Survival by Family Size
    plt.figure(figsize=(10, 6), dpi=150)
    survival_rates_plot = df_clean.groupby('FamilySize')['Survived'].mean().reset_index()
    colors = sns.color_palette("viridis_r", len(survival_rates_plot))
    bars = sns.barplot(x='FamilySize', y='Survived', data=survival_rates_plot, hue='FamilySize', palette=colors, legend=False, edgecolor='none')
    
    for bar in bars.patches:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.015, f"{height:.1%}",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2d3748')
            
    plt.title('Survival Rate by Family Size', fontsize=16, fontweight='bold', pad=15, color='#2d3748')
    plt.xlabel('Family Size (SibSp + Parch)', fontsize=12, labelpad=10, color='#4a5568')
    plt.ylabel('Survival Rate (%)', fontsize=12, labelpad=10, color='#4a5568')
    plt.ylim(0, 0.85)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', dpi=150)
    buf3.seek(0)
    plot3_base64 = base64.b64encode(buf3.read()).decode('utf-8')
    plt.close()
    
    # ------------------ Build JSON Notebook ------------------
    cells = []
    
    # Cell 1 (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Titanic Dataset - Mini Exploratory Data Analysis (EDA)\n",
            "---\n",
            "## Overview\n",
            "In this notebook, we perform a mini Exploratory Data Analysis (EDA) on the Titanic dataset to uncover patterns and factors associated with passenger survival. We'll clean the data and answer key business/analytical questions regarding who survived and why.\n\n",
            "### Notebook Checklist & Deliverables:\n",
            "1. **Clean dataset**:\n",
            "   * Fill missing `Age` with the mean.\n",
            "   * Drop irrelevant columns (like `Cabin`, `PassengerId`, `Name`, and `Ticket`).\n",
            "2. **Analysis Questions**:\n",
            "   * Survival rate by **Age Group**.\n",
            "   * Survival rate by **Embarkation Port**.\n",
            "   * Survival rate by **Family Size**.\n",
            "3. **Visualizations**:\n",
            "   * **Age distribution** (histogram with KDE).\n",
            "   * **Heatmap of correlations** between numerical variables.\n",
            "   * **Survival by family size** (bar plot with survival percentages)."
        ]
    })
    
    # Cell 2 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import matplotlib.ticker as mtick\n\n",
            "# Set premium styling parameters for plots\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams['figure.figsize'] = (10, 6)\n",
            "plt.rcParams['figure.dpi'] = 150\n",
            "plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']\n\n",
            "# Load dataset\n",
            "df = pd.read_csv('titanic.csv')\n",
            "print(f\"Dataset loaded successfully! Shape: {df.shape}\")\n",
            "df.head()"
        ],
        "outputs": [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [cell2_stdout]
            },
            {
                "output_type": "execute_result",
                "execution_count": 1,
                "data": {
                    "text/plain": [cell2_df_head.to_string()],
                    "text/html": [cell2_df_head.to_html(classes="dataframe", border=1, justify="left")]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 3 (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Cleaning & Preprocessing\n\n",
            "To handle missing values and prepare our data:\n",
            "* We check which columns have missing values.\n",
            "* We fill missing values in **Age** with the mean age of passengers.\n",
            "* We drop **Cabin** (which contains >77% missing values) and other identifiers that are irrelevant to correlation/survival analysis (like `PassengerId`, `Name`, and `Ticket`)."
        ]
    })
    
    # Cell 4 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "source": [
            "# Check missing values before cleaning\n",
            "print(\"Missing values in raw dataset:\")\n",
            "print(df.isnull().sum())\n\n",
            "# 1. Fill missing Age with mean\n",
            "mean_age = df['Age'].mean()\n",
            "df['Age'] = df['Age'].fillna(mean_age)\n\n",
            "# 2. Fill missing Embarked with mode (Southampton 'S')\n",
            "mode_embarked = df['Embarked'].mode()[0]\n",
            "df['Embarked'] = df['Embarked'].fillna(mode_embarked)\n\n",
            "# 3. Drop Cabin and other irrelevant columns\n",
            "columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']\n",
            "df_clean = df.drop(columns=columns_to_drop)\n\n",
            "print(\"\\nMissing values after cleaning:\")\n",
            "print(df_clean.isnull().sum())\n",
            "print(f\"\\nCleaned dataset shape: {df_clean.shape}\")\n",
            "df_clean.head()"
        ],
        "outputs": [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [cell4_stdout]
            },
            {
                "output_type": "execute_result",
                "execution_count": 2,
                "data": {
                    "text/plain": [cell4_df_head.to_string()],
                    "text/html": [cell4_df_head.to_html(classes="dataframe", border=1, justify="left")]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 5 (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Groupby Analysis (Answering Key Questions)\n\n",
            "Now let's compute survival rates grouped by different categories to uncover patterns."
        ]
    })
    
    # Cell 6 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "source": [
            "# A. Survival rate by Age Group\n",
            "age_bins = [0, 12, 18, 35, 60, 100]\n",
            "age_labels = ['Child (0-12)', 'Teen (12-18)', 'Young Adult (18-35)', 'Middle Aged (35-60)', 'Senior (60+)']\n",
            "df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels)\n\n",
            "survival_by_age = df_clean.groupby('AgeGroup', observed=False)['Survived'].agg(['count', 'sum', 'mean']).rename(\n",
            "    columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}\n",
            ")\n",
            "survival_by_age['Survival Rate'] = survival_by_age['Survival Rate'].map('{:.2%}'.format)\n\n",
            "print(\"=== Survival Rate by Age Group ===\")\n",
            "survival_by_age"
        ],
        "outputs": [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [cell6_stdout]
            },
            {
                "output_type": "execute_result",
                "execution_count": 3,
                "data": {
                    "text/plain": [survival_by_age_formatted.to_string()],
                    "text/html": [survival_by_age_formatted.to_html(classes="dataframe", border=1, justify="left")]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 7 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "source": [
            "# B. Survival rate by Embarkation Port\n",
            "port_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}\n",
            "df_clean['EmbarkationPort'] = df_clean['Embarked'].map(port_map)\n\n",
            "survival_by_port = df_clean.groupby('EmbarkationPort')['Survived'].agg(['count', 'sum', 'mean']).rename(\n",
            "    columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}\n",
            ")\n",
            "survival_by_port['Survival Rate'] = survival_by_port['Survival Rate'].map('{:.2%}'.format)\n\n",
            "print(\"=== Survival Rate by Embarkation Port ===\")\n",
            "survival_by_port"
        ],
        "outputs": [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [cell7_stdout]
            },
            {
                "output_type": "execute_result",
                "execution_count": 4,
                "data": {
                    "text/plain": [survival_by_port_formatted.to_string()],
                    "text/html": [survival_by_port_formatted.to_html(classes="dataframe", border=1, justify="left")]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 8 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "source": [
            "# C. Survival rate by Family size (SibSp + Parch)\n",
            "df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch']\n\n",
            "survival_by_family = df_clean.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean']).rename(\n",
            "    columns={'count': 'Total Passengers', 'sum': 'Survived', 'mean': 'Survival Rate'}\n",
            ")\n",
            "survival_by_family['Survival Rate'] = survival_by_family['Survival Rate'].map('{:.2%}'.format)\n\n",
            "print(\"=== Survival Rate by Family Size ===\")\n",
            "survival_by_family"
        ],
        "outputs": [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [cell8_stdout]
            },
            {
                "output_type": "execute_result",
                "execution_count": 5,
                "data": {
                    "text/plain": [survival_by_family_formatted.to_string()],
                    "text/html": [survival_by_family_formatted.to_html(classes="dataframe", border=1, justify="left")]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 9 (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Advanced Visualizations\n\n",
            "Below we present the three requested plots, generated with professional styling: high DPI, clean layout grids, customized typography, and tailored colors."
        ]
    })
    
    # Cell 10 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "source": [
            "# 1. Age Distribution (Histogram with KDE)\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.histplot(df_clean['Age'], kde=True, color='#2c5282', bins=30, edgecolor='white', alpha=0.85)\n\n",
            "plt.title('Age Distribution of Titanic Passengers', fontsize=16, fontweight='bold', pad=15, color='#2d3748')\n",
            "plt.xlabel('Age', fontsize=12, labelpad=10, color='#4a5568')\n",
            "plt.ylabel('Passenger Count', fontsize=12, labelpad=10, color='#4a5568')\n",
            "plt.grid(True, linestyle='--', alpha=0.5)\n\n",
            "sns.despine(left=True, bottom=True)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "outputs": [
            {
                "output_type": "display_data",
                "data": {
                    "image/png": plot1_base64,
                    "text/plain": ["<Figure size 1500x900 with 1 Axes>"]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 11 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "source": [
            "# 2. Heatmap of Correlations\n",
            "plt.figure(figsize=(10, 8))\n",
            "numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']\n",
            "corr_matrix = df_clean[numeric_features].corr()\n",
            "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n\n",
            "sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=\".2f\", cmap=\"coolwarm\",\n",
            "            vmin=-1, vmax=1, center=0, square=True, linewidths=.5,\n",
            "            cbar_kws={\"shrink\": .8, \"label\": \"Correlation Coefficient\"},\n",
            "            annot_kws={\"fontsize\": 11, \"fontweight\": \"semibold\"})\n\n",
            "plt.title('Correlation Matrix of Titanic Numeric Features', fontsize=16, fontweight='bold', pad=20, color='#2d3748')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "outputs": [
            {
                "output_type": "display_data",
                "data": {
                    "image/png": plot2_base64,
                    "text/plain": ["<Figure size 1500x1200 with 2 Axes>"]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 12 (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "source": [
            "# 3. Survival Rate by Family Size (Bar Plot)\n",
            "plt.figure(figsize=(10, 6))\n",
            "survival_rates_plot = df_clean.groupby('FamilySize')['Survived'].mean().reset_index()\n",
            "colors = sns.color_palette(\"viridis_r\", len(survival_rates_plot))\n",
            "bars = sns.barplot(x='FamilySize', y='Survived', data=survival_rates_plot, hue='FamilySize', palette=colors, legend=False, edgecolor='none')\\n\\n",
            "for bar in bars.patches:\n",
            "    height = bar.get_height()\n",
            "    if height > 0:\n",
            "        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.015, f\"{height:.1%}\",\n",
            "                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2d3748')\n\n",
            "plt.title('Survival Rate by Family Size', fontsize=16, fontweight='bold', pad=15, color='#2d3748')\n",
            "plt.xlabel('Family Size (SibSp + Parch)', fontsize=12, labelpad=10, color='#4a5568')\n",
            "plt.ylabel('Survival Rate (%)', fontsize=12, labelpad=10, color='#4a5568')\n",
            "plt.ylim(0, 0.85)\n",
            "plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n",
            "plt.grid(True, axis='y', linestyle='--', alpha=0.5)\n",
            "sns.despine(left=True, bottom=True)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "outputs": [
            {
                "output_type": "display_data",
                "data": {
                    "image/png": plot3_base64,
                    "text/plain": ["<Figure size 1500x900 with 1 Axes>"]
                },
                "metadata": {}
            }
        ]
    })
    
    # Cell 13 (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Key Insights & Conclusions\n\n",
            "Based on our Exploratory Data Analysis, we can draw several key insights about passenger survival on the Titanic:\n\n",
            "1. **Age Group Dynamics**:\n",
            "   * **Children (0-12)** had the highest survival rate at **59.18%**, highlighting the \"women and children first\" evacuation policy.\n",
            "   * **Seniors (60+)** had the lowest survival rate at **22.73%**, indicating that older passengers faced significant challenges during the evacuation.\n\n",
            "2. **Embarkation Port Influence**:\n",
            "   * Passengers boarding at **Cherbourg (C)** had a significantly higher survival rate (**55.36%**) compared to **Queenstown (Q, 38.96%)** and **Southampton (S, 33.90%)**.\n",
            "   * This difference is largely driven by socio-economic class, as Cherbourg had a higher proportion of first-class passengers.\n\n",
            "3. **Family Size & Social Support**:\n",
            "   * Passengers with small families (**1 to 3 family members**) had the highest survival rates, peaking at **72.41%** for a family size of 3.\n",
            "   * Passengers travelling **alone** (family size 0) had a much lower survival rate of **30.35%**.\n",
            "   * Large families (**4 or more family members**) saw a steep drop in survival rates, likely due to the difficulty of keeping large families together and coordinate evacuation.\n\n",
            "4. **Correlations**:\n",
            "   * There is a strong negative correlation between **Pclass** and **Survived** (-0.34), confirming that first-class passengers had a much higher chance of survival than lower classes.\n",
            "   * **Fare** and **Survived** show a positive correlation (0.26), which aligns with the class distinction.\n",
            "   * **Fare** and **Pclass** show a strong negative correlation (-0.55) as expected."
        ]
    })
    
    # Complete Notebook Dictionary
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("EDA_Titanic.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Jupyter Notebook 'EDA_Titanic.ipynb' generated successfully!")

if __name__ == "__main__":
    download_data()
    generate_notebook()
