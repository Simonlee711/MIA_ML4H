import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    diagnosis = pd.read_csv('diagnosis.csv')
    medrecon = pd.read_csv('medrecon.csv')
    edstays = pd.read_csv('edstays.csv')
    triage = pd.read_csv('triage.csv')
    pyxis = pd.read_csv('pyxis.csv')
    vitalsign = pd.read_csv('vitalsign.csv')

    #merge
    edstays_triage = pd.merge(edstays, triage, on = ['stay_id', 'subject_id'], how = 'left')
    edstays_triage_diagnosis = pd.merge(edstays_triage, diagnosis, on = ['stay_id', 'subject_id'], how = 'left')
    edstays_triage_diagnosis_medrecon = pd.merge(edstays_triage_diagnosis, medrecon, on = ['stay_id', 'subject_id'], how = 'left')

    
    #drop columns, na, duplicates
    df = edstays_triage_diagnosis_medrecon.copy()
    df = df.drop(columns = ['hadm_id', 'icd_title', 'name', 'etcdescription', 'charttime'])
    df = df.dropna()
    df = df.drop_duplicates()
    
    #only keep home and admitted
    df = df[df['disposition'].isin(['HOME', 'ADMITTED'])]
    #combine chiefcomplaints (FEVER + fever)
    df.loc[:, 'chiefcomplaint'] = df['chiefcomplaint'].str.lower().str.replace(' ', '', regex=False)

    #ndc and gsn = 0 means missing
    df = df[df['ndc'] != 0]
    df = df[df['gsn'] != 0]



    #only keep pain when rated as a number (only keep up to 13 but maybe limit to 10)
    df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
    df = df.dropna(subset=['pain'])
    df['pain'] = df['pain'].astype(int)
    df = df[df['pain'] <= 13]



    #only keep top 20 most frequent for the following columns
    columns_to_filter = ['chiefcomplaint', 'icd_code', 'gsn', 'ndc', 'etccode']
    for col in columns_to_filter:
        top_20_categories = df[col].value_counts().head(20).index
        df = df[df[col].isin(top_20_categories)] 
        

    #change intime and outtime to total time in ed    
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    df['totaltime'] = df['outtime'] - df['intime']
    df = df.drop(columns = ['intime', 'outtime'])
    df['totaltime'] = df['totaltime'].dt.total_seconds() / 3600

    
    #reduce race to 5 categories
    def categorize_race(race):
        
        race_lower = str(race).lower()
        if 'white' in race_lower:
            return 'WHITE'
        elif 'black' in race_lower:
            return 'BLACK/AFRICAN AMERICAN'
        elif 'hispanic' in race_lower:
            return 'HISPANIC/LATINO'
        elif 'asian' in race_lower:
            return 'ASIAN'
        else:
            return 'OTHER/UNKNOWN'

    df['race'] = df['race'].apply(categorize_race)


    
    df_label_encoded = df.copy()


    #label encode categorical variables
    cat_cols = ['gender', 'race', 'arrival_transport', 'disposition','icd_version']

    for col in cat_cols:
        df_label_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

    #one hot encode other categorical variables
    df_label_encoded = pd.get_dummies(df_label_encoded, columns=['chiefcomplaint', 'icd_code', 'etccode', 'gsn', 'ndc'], dtype=int)


    #Scale numerical columns
    numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'totaltime']  
    scaler = StandardScaler()
    df_label_encoded[numerical_cols] = scaler.fit_transform(df_label_encoded[numerical_cols])
        
    return df_label_encoded