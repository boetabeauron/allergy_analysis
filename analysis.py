import pandas as pd
import sketch
import numpy as np
import os

os.environ["SKETCH_MAX_COLUMNS"] = "100"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import warnings

warnings.filterwarnings("ignore")
import pycountry
from datetime import timedelta
import hvplot.pandas
import holoviews as hv

patients_df = pd.read_csv("./patients.csv")
conditions_df = pd.read_csv("./conditions.csv")
medication_df = pd.read_csv("./medications.csv")
payer_df = pd.read_csv("./payers.csv")
provider_df = pd.read_csv("./providers.csv")
organization_df = pd.read_csv("./organizations.csv")

# Remove leading/trailing spaces
patients_df.columns = patients_df.columns.str.strip()

# Update Birthdate & Deathdate to datetime.
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["DEATHDATE"] = pd.to_datetime(patients_df["DEATHDATE"])

#  Rename Id, LAT & LON to stadardize the dataframe
patients_df = patients_df.rename(
    columns={"Id": "patient_id", "LAT": "latitude", "LON": "longitude"}
)
# Convert column name to lower_case
patients_df.columns = patients_df.columns.str.lower()

# Remove columns that's not required
patients_df = patients_df[
    [
        "patient_id",
        "birthdate",
        "deathdate",
        "ssn",
        "drivers",
        "passport",
        "prefix",
        "first",
        "last",
        "marital",
        "maiden",
        "race",
        "ethnicity",
        "gender",
        "birthplace",
        "city",
        "state",
        "county",
        "latitude",
        "longitude",
        "healthcare_expenses",
        "healthcare_coverage",
    ]
]

# Conver the dataframe to lowercase
patients_df = patients_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
# Standardize the gender where from 'm' to 'male' & 'f' to 'female'
patients_df["gender"] = patients_df["gender"].replace({"m": "male", "f": "female"})

# Standardize the gender where from 'm' to 'married' & 's' to 'single'
patients_df["marital"] = patients_df["marital"].replace({"m": "married", "s": "single"})

# Update the marital status where prefix is Ms to Single where marital is Null
patients_df.loc[
    (patients_df["prefix"] == "ms.") & (patients_df["marital"].isnull()), "marital"
] = "single"

# Calculate the age of the patients
today = pd.to_datetime("today")
patients_df["age_today"] = (today - patients_df["birthdate"]) // pd.Timedelta(
    days=365.25
)
patients_df["age_today"] = patients_df["age_today"].astype(int)

# Calculate the age of the patient at death
patients_df["death_age"] = (
    patients_df["deathdate"] - patients_df["birthdate"]
).dt.days / 365.25
patients_df["death_age"] = patients_df["death_age"].fillna(value=0)
patients_df["death_age"] = patients_df["death_age"].astype(int)

# Replace age_today with death_age where death_date exists
patients_df.loc[patients_df["deathdate"].notnull(), "age_today"] = patients_df[
    "death_age"
]

# Create new column actual_age
patients_df["patient_age"] = patients_df["age_today"]

# Replace na values in prefix column with 'mr.' if gender is male and 'ms.' if gender is female
patients_df.loc[patients_df["gender"] == "male", "prefix"] = patients_df[
    "prefix"
].fillna("mr.")
patients_df.loc[patients_df["gender"] == "female", "prefix"] = patients_df[
    "prefix"
].fillna("ms.")

# Create a pivot table with the last name as the index and the count of unique patient IDs as the values
last_name_counts = pd.pivot_table(
    patients_df, index="last", values="patient_id", aggfunc=pd.Series.nunique
)

# Rename the column to 'unique_count'
last_name_counts.rename(columns={"patient_id": "unique_count"}, inplace=True)

# Sort the pivot table by the unique count in descending order
last_name_counts.sort_values(by="unique_count", ascending=False, inplace=True)
last_name_counts.head()

patients_df.loc[874, "marital"] = "married"
patients_df.loc[194, "marital"] = "mx"

patients_df.loc[
    (patients_df["marital"].isna()) & (patients_df["age_today"] < 16), "marital"
] = "single"
patients_df.loc[
    (patients_df["marital"].isna()) & (patients_df["age_today"] >= 16), "marital"
] = "mx"
patients_df = patients_df[
    [
        "patient_id",
        "birthdate",
        "deathdate",
        "ssn",
        "prefix",
        "marital",
        "race",
        "ethnicity",
        "gender",
        "birthplace",
        "city",
        "state",
        "county",
        "latitude",
        "longitude",
        "healthcare_expenses",
        "healthcare_coverage",
        "age_today",
    ]
]

# Add a status column
patients_df["status"] = ""

# Check if deathdate is null and add status accordingly
patients_df.loc[patients_df["deathdate"].isna(), "status"] = "alive"
patients_df.loc[~patients_df["deathdate"].isna(), "status"] = "deceased"

# Print head of dataframe to check results
patients_df.head(2)

# This is the ALLERGY DATASET
allergy_df = pd.read_csv("./allergies.csv")
allergy_df.columns = allergy_df.columns.str.lower()
allergy_df = allergy_df.rename(
    columns={"patient": "patient_id", "encounter": "encounter_id"}
)
allergy_df = allergy_df.drop_duplicates()
allergy_df.head()

refined_allergy_df = allergy_df[["patient_id", "encounter_id"]]
refined_allergy_df = refined_allergy_df.drop_duplicates()
refined_allergy_df["encounter_reason"] = "allergy"
refined_allergy_df.head()


encounter_df = pd.read_csv("./encounters.csv")
encounter_df.columns = encounter_df.columns.str.lower()
encounter_df = encounter_df.rename(
    columns={
        "id": "encounter_id",
        "patient": "patient_id",
    }
)

# Merge the refined_dataframe to the encounter_df to determine which encounter was allergy related.

encounter_merge_df = pd.merge(
    encounter_df, refined_allergy_df, on="encounter_id", how="right"
)

encounter_merge_df = encounter_merge_df[
    [
        "encounter_id",
        "start",
        "stop",
        "patient_id_x",
        "organization",
        "provider",
        "payer",
        "encounterclass",
        "code",
        "description",
        "base_encounter_cost",
        "total_claim_cost",
        "payer_coverage",
        "reasoncode",
        "reasondescription",
        "encounter_reason",
    ]
]

encounter_merge_df = encounter_merge_df[
    encounter_merge_df["encounter_reason"] == "allergy"
]
encounter_merge_df = encounter_merge_df.rename(columns={"patient_id_x": "patient_id"})

# Merge the encounter_merge_df with patients_df on patient_id
full_encounter_merged_df = encounter_merge_df.merge(patients_df, on="patient_id")

# Print summary statistics and descriptive data of the new dataframe
full_encounter_merged_df = full_encounter_merged_df[
    [
        "encounter_id",
        "start",
        "stop",
        "patient_id",
        "organization",
        "provider",
        "payer",
        "encounterclass",
        "code",
        "base_encounter_cost",
        "total_claim_cost",
        "payer_coverage",
        "healthcare_expenses",
        "healthcare_coverage",
        "encounter_reason",
        "birthdate",
        "deathdate",
        "ssn",
        "prefix",
        "marital",
        "race",
        "ethnicity",
        "gender",
        "birthplace",
        "city",
        "state",
        "county",
        "latitude",
        "longitude",
        "age_today",
        "status",
    ]
]


# Define a function to split location and country code
def split_location_country(location):
    # Split the location string by the last space to separate the country code
    parts = location.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return location, None


# Apply the function to split 'location' into 'location' and 'country'
full_encounter_merged_df[["birthplace", "country"]] = full_encounter_merged_df[
    "birthplace"
].apply(lambda x: pd.Series(split_location_country(x)))


# Define a function to convert country codes to full country names
def convert_country_code_to_name(code):
    try:
        return pycountry.countries.get(alpha_2=code).name
    except:
        return None


# Use the function to convert the country codes
full_encounter_merged_df["country"] = full_encounter_merged_df["country"].apply(
    convert_country_code_to_name
)

# Display the updated DataFrame


# Remove timezone information if present
full_encounter_merged_df["start"] = pd.to_datetime(
    full_encounter_merged_df["start"]
).dt.tz_localize(None)
full_encounter_merged_df["stop"] = pd.to_datetime(
    full_encounter_merged_df["stop"]
).dt.tz_localize(None)

# Convert 'code' column to string
full_encounter_merged_df["code"] = full_encounter_merged_df["code"].astype(str)

# Calculate the duration between 'stop' and 'start'
full_encounter_merged_df["days_in_hospital"] = (
    full_encounter_merged_df["stop"] - full_encounter_merged_df["start"]
)

# Create a new column showing the number of days as integers
full_encounter_merged_df["days_in_hospital_int"] = full_encounter_merged_df[
    "days_in_hospital"
].dt.days

# Convert timedelta to total minutes
full_encounter_merged_df["service_in_minutes"] = (
    full_encounter_merged_df["days_in_hospital"].dt.total_seconds() // 60
)

# Convert service_in_minutes to integer type
full_encounter_merged_df["service_in_minutes"] = full_encounter_merged_df[
    "service_in_minutes"
].astype(int)

# Merge the organization_df to fulldf.
full_encounter_merged_df = pd.merge(
    full_encounter_merged_df, organization_df, left_on="organization", right_on="Id"
)
full_encounter_merged_df = pd.merge(
    full_encounter_merged_df, provider_df, left_on="provider", right_on="Id"
)
full_encounter_merged_df = pd.merge(
    full_encounter_merged_df, payer_df, left_on="payer", right_on="Id"
)

full_encounter_merged_df = full_encounter_merged_df[
    [
        "encounter_id",
        "start",
        "stop",
        "patient_id",
        "encounterclass",
        "code",
        "base_encounter_cost",
        "total_claim_cost",
        "payer_coverage",
        "healthcare_expenses",
        "healthcare_coverage",
        "encounter_reason",
        "birthdate",
        "deathdate",
        "ssn",
        "prefix",
        "marital",
        "race",
        "ethnicity",
        "gender",
        "birthplace",
        "city",
        "state",
        "county",
        "latitude",
        "longitude",
        "age_today",
        "status",
        "country",
        "days_in_hospital",
        "days_in_hospital_int",
        "service_in_minutes",
        "NAME_x",
        "NAME_y",
        "GENDER",
        "NAME",
    ]
]

full_encounter_merged_df = full_encounter_merged_df.rename(
    columns={
        "NAME_x": "medical_center",
        "NAME_y": "practicioner_name",
        "NAME": "medical_aid_scheme",
        "GENDER": "practicioner_gender",
    }
)

# Rename to Male and Female in practicioner_gender column
full_encounter_merged_df["practicioner_gender"] = full_encounter_merged_df[
    "practicioner_gender"
].replace({"M": "male", "F": "female"})

# Make the dataframe all in lower case
full_encounter_merged_df = full_encounter_merged_df.apply(
    lambda x: x.str.lower() if x.dtype == "object" else x
)

# Edit the medical aid scheme name by cleaning up inconsistency
full_encounter_merged_df["medical_aid_scheme"] = full_encounter_merged_df[
    "medical_aid_scheme"
].replace({"unitedhealthcare": "united health care", "no_insurance": "no insurance"})

# Edit the country name venezuela for standarization
full_encounter_merged_df["country"] = full_encounter_merged_df["country"].replace(
    {"venezuela, bolivarian republic of": "venezuela"}
)

full_encounter_merged_df.head(1)

full_encounter_merged_df["medical_center"] = full_encounter_merged_df[
    "medical_center"
].replace(
    {
        "heywood hospital -": "heywood hospital",
        "beth israel deaconess hospital-milton inc": "beth israel deaconess hospital-milton",
        "mercy medical ctr": "mercy medical center",
        "southcoast hospital group  inc": "southcoast hospital",
        "harrington memorial hospital-1": "harrington memorial hospital",
        "shriners' hospital for children - boston  the": "shriners' hospital for children - boston",
        "north shore medical center -": "north shore medical center",
        "umass memorial medical center inc": "umass memorial medical center",
        "cooley dickinson hospital inc the": "cooley dickinson hospital inc",
        "massachusetts eye and ear infirmary -": "massachusetts eye and ear infirmary",
        "berkshire medical center inc - 1:": "berkshire medical center inc",
        "berkshire medical center inc - 1": "berkshire medical center inc",
    }
)
df = full_encounter_merged_df
df["encounter_year"] = df["start"].dt.year
df.head(1)


cases_over_time = df.groupby(by="encounter_year").size().reset_index(name="cases")
cases_over_time.hvplot.line(
    x="encounter_year",
    y="cases",
    legend="top",
    height=600,
    width=800,
    title="Cases over time",
)

df = df[(df["encounter_year"] >= 2000)]
# Group by gender and count occurrences
gender_df = df.groupby(by="gender").size().reset_index(name="count")
gender_df.hvplot.bar(
    x="gender", y="count", width=500, height=450, grid=True, legend=False
).opts(color="gender", cmap=["pink", "blue"])

df.hvplot.hist("age_today")

df.hvplot.box(
    y="age_today",
    by="gender",
    height=400,
    width=400,
    legend=False,
    grid=True,
).opts(box_color="gender", cmap=["pink", "blue"])
