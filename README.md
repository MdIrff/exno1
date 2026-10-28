# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output

Import libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
```
Load the datasets
```
files = [
    "Data_set.csv",
    "heights.csv",
    "iris.csv",
    "Loan_data.csv",
    "SAMPLEIDS.csv"
]

datasets = {}

for file in files:
    datasets[file] = pd.read_csv(file)
    print(f"\n{file}")
    print(datasets[file].head())
```

<img width="590" height="520" alt="image" src="https://github.com/user-attachments/assets/a83d0441-8065-4d92-8663-7e192e5e990b" />

Explore the datasets

```
for name, df in datasets.items():
    print("\nDataset:", name)
    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nStatistical summary:")
    display(df.describe(include="all"))
```

<img width="341" height="504" alt="image" src="https://github.com/user-attachments/assets/a9d1d40c-21e3-4299-85f9-e0dff97fab33" />
<img width="351" height="559" alt="image" src="https://github.com/user-attachments/assets/e493c6a1-d366-47ae-bb13-51853b9a8af2" />
<img width="404" height="387" alt="image" src="https://github.com/user-attachments/assets/ea2273ef-9053-4e58-b5f7-8e982cb2edf9" />
<img width="440" height="440" alt="image" src="https://github.com/user-attachments/assets/07290f78-e42f-4533-9dc3-7b0ffc1e9b01" />

Perform data cleaning
```
cleaned_datasets = {}

for name, df in datasets.items():
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert empty strings to missing values
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Fill missing numerical values with the median
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with the mode
    categorical_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    cleaned_datasets[name] = df

    print(name, "cleaned shape:", df.shape)

```

<img width="287" height="84" alt="image" src="https://github.com/user-attachments/assets/4621710c-d01b-4c2d-891e-08f7db418f6d" />


Detect outliers using the IQR method

```
outlier_reports = {}

for name, df in cleaned_datasets.items():

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    report = []

    for col in numeric_cols:
        values = df[col].dropna()

        if values.empty or values.nunique() < 2:
            continue

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = ((df[col] < lower) | (df[col] > upper)).sum()

        report.append({
            "Column": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outliers": int(count)
        })

    outlier_reports[name] = pd.DataFrame(report)

    print("\nOutlier report:", name)
    display(outlier_reports[name])
```

<img width="389" height="463" alt="image" src="https://github.com/user-attachments/assets/f36fa5fb-8977-4041-82fe-3ad797838e3b" />


Visualize outliers using boxplots
```
for name, df in cleaned_datasets.items():

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_cols:
        continue

    df[numeric_cols].plot(
        kind="box",
        subplots=True,
        layout=(-1, 3),
        figsize=(12, max(4, len(numeric_cols) * 2)),
        sharex=False,
        sharey=False
    )

    plt.suptitle(f"Outlier Detection: {name}")
    plt.tight_layout()
    plt.show()
```
<img width="483" height="380" alt="image" src="https://github.com/user-attachments/assets/147b7275-0d52-4c73-944e-47aacb1c1084" />
<img width="436" height="185" alt="image" src="https://github.com/user-attachments/assets/9e8e2114-5982-409e-ae5e-665172f2d7ae" />
<img width="482" height="301" alt="image" src="https://github.com/user-attachments/assets/3a173a14-e187-480a-8156-d4fc6139c9ac" />
<img width="278" height="319" alt="image" src="https://github.com/user-attachments/assets/b21f05db-bf28-48bc-bce7-a445c1b32140" />
<img width="338" height="439" alt="image" src="https://github.com/user-attachments/assets/d27e852a-7479-4fe2-9a6f-1aaa80177844" />

Remove outliers
```
outlier_free_datasets = {}

for name, df in cleaned_datasets.items():

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Keep rows by checking IQR bounds for each numeric column
    keep = pd.Series(True, index=df.index)

    for col in numeric_cols:
        values = df[col].dropna()

        if values.empty or values.nunique() < 2:
            continue

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        keep &= df[col].between(lower, upper) | df[col].isna()

    result = df.loc[keep].copy()
    outlier_free_datasets[name] = result

    print(f"\n{name}")
    print("Rows before removal:", len(df))
    print("Rows after removal:", len(result))
    print("Rows removed:", len(df) - len(result))
```

<img width="208" height="248" alt="image" src="https://github.com/user-attachments/assets/178b6ba8-524c-4ec6-85c2-fc3ead1acf1d" />

Save the cleaned datasets and reports
```
os.makedirs("cleaned_data", exist_ok=True)

for name, df in outlier_free_datasets.items():
    output_name = os.path.splitext(name)[0] + "_cleaned.csv"
    df.to_csv(os.path.join("cleaned_data", output_name), index=False)

# Combine the outlier reports
report_list = []

for name, report in outlier_reports.items():
    if not report.empty:
        temp = report.copy()
        temp.insert(0, "Dataset", name)
        report_list.append(temp)

if report_list:
    final_report = pd.concat(report_list, ignore_index=True)
    final_report.to_csv(
        "cleaned_data/outlier_report.csv",
        index=False
    )
    display(final_report)

print("All available cleaned datasets and reports have been saved.")
```

<img width="533" height="375" alt="image" src="https://github.com/user-attachments/assets/37544f8c-856f-43fd-8f0e-910e37d48597" />


Summary

In this project, I cleaned five datasets (Data_set, heights, iris, Loan_data and SAMPLEIDS) using Python and pandas.

First, I loaded each dataset and checked its size, data types, missing values and duplicates. Then I cleaned the data by removing duplicate rows and empty columns, and filling missing values: numbers with the median and text with the most frequent value (mode).

Next, I found outliers using the IQR method, which flags values that fall far outside the middle range of the data. I showed them using boxplots and then removed the rows containing outliers.

Finally, I saved the cleaned datasets and an outlier report as CSV files.


           
# Result

The datasets are inspected and cleaned by handling missing values and duplicates. Potential outliers are detected using the IQR method, visualized with boxplots, and removed. The processed datasets and outlier reports are saved for further analysis.
