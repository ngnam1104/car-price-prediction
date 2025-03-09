import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import datetime

def get_re_group(r, txt):
    match = re.search(r, txt)

    if match:
        return float(match.groups()[0])
    else:
        return np.nan
    
def determine_fuel_type(engine):
    if pd.isna(engine):
        return None
    if re.search(r'Electric|Battery|kW|charge|kw', engine, re.IGNORECASE):
        return 'Electric'
    if re.search(r'Diesel', engine, re.IGNORECASE):
        return 'Diesel'
    if re.search(r'Flex Fuel|E85', engine, re.IGNORECASE):
        return 'E85 Flex Fuel'
    if re.search(r'Gasoline|Standard|Turbo|Liter|GDI|MPFI|PDI', engine, re.IGNORECASE):
        return 'Gasoline'
    return None

def handle_some_missing_fuel_type(data):
    mask = ((data['fuel_type'] != 'Diesel') & (data['fuel_type'] != 'Electric') & (data['fuel_type'] != 'E85 Flex Fuel') & 
            (data['fuel_type'] != 'Gasoline') & (data['fuel_type'] != 'Hybrid') & (data['fuel_type'] != 'Plug-In Hybrid'))
    data.loc[mask, 'fuel_type'] = data.loc[mask, 'engine'].apply(determine_fuel_type)
    return data

def preprocess(data):
    data['horse_power'] = data['engine'].apply(lambda en: get_re_group(r'(\d+\.\d+)HP', en))
    data['engine_size'] = data['engine'].apply(lambda en: get_re_group(r'(\d+\.\d+)L', en))
    data['cylinder'] = data['engine'].apply(lambda en: get_re_group(r'(\d+) Cylinder', en))

    data = handle_some_missing_fuel_type(data)

    data['transmission_speed'] = data['transmission'].apply(lambda trans: get_re_group(r'(\d+)', trans))

    data['transmission_type'] = np.nan
    data['transmission_type'] = data['transmission_type'].astype('object')
    data.loc[data['transmission'].str.contains(r'Manual|MT|M/T', na=False, case=False), 'transmission_type'] = 'M/T'
    data.loc[data['transmission'].str.contains(r'A/T|Automatic|AT', na=False, case=False), 'transmission_type'] = 'A/T'
    data.loc[data['transmission'].str.contains(r'CVT|Variable', na=False, case=False), 'transmission_type'] = 'CVT'
    data.loc[data['transmission'].str.contains(r'DCT|Dual Shift Mode', na=False, case=False), 'transmission_type'] = 'DCT'
    data.loc[data['transmission'].str.contains(r'Fixed Gear', na=False, case=False), 'transmission_type'] = 'Fixed Gear'
    data.loc[data['transmission'].str.contains(r'Electronically Controlled Automatic', na=False, case=False), 'transmission_type'] = 'Electronically Controlled'
    data.loc[data['transmission'].str.contains(r'Auto-Shift|AT/MT', na=False, case=False), 'transmission_type'] = 'Auto-Shift'
    data.loc[data['transmission'].str.contains(r'Overdrive', na=False, case=False), 'transmission_type'] = 'Overdrive'

    data['car_age'] = datetime.datetime.now().year - data['model_year']
    data.loc[data['car_age'] == 0, 'car_age'] = 1
    data['yearly_mileage'] = data['milage'] / data['car_age']

    data['model_class'] = np.nan
    data['model_class'] = data['model_class'].astype('object')
    data.loc[data['model'].str.contains('S|L|Base', na=False, case=False), 'model_class'] = 'Base'
    data.loc[data['model'].str.contains('SE|SX|SRT|GT', na=False, case=False), 'model_class'] = 'Mid-Range'
    data.loc[data['model'].str.contains('EX|SXT', na=False, case=False), 'model_class'] = 'Performance/Sport'
    data.loc[data['model'].str.contains('EX-L|LX|LE|SEL|Platinum|Premium|Limited', na=False, case=False), 'model_class'] = 'Luxury'

    luxury_brands = 'Mercedes-Benz|BMW|Audi|Porsche|Lexus|Cadillac|Jaguar|Bentley|Maserati|Lamborghini|Rolls-Royce|Ferrari|McLaren|Aston|Martin|Lucid|Lotus|Karma|Bugatti|Maybach'
    premium_brands = 'Acura|Infiniti|Genesis|Volvo|Lincoln|Land|Rover'
    mainstream_brands = 'Ford|Chevrolet|Toyota|Jeep|RAM|Nissan|Tesla|GMC|Dodge|Mazda|Kia|Subaru|Honda|Hyundai|Volkswagen|Buick|Chrysler|Mitsubishi|Polestar|Rivian'
    economy_brands = 'MINI|Fiat|Saab|Suzuki|smart'
    supercar_brands = 'Lamborghini|Ferrari|McLaren|Bugatti'

    data['brand_cat'] = np.nan
    data['brand_cat'] = data['brand_cat'].astype('object')
    data.loc[data['brand'].str.contains(luxury_brands, na=False, case=False), 'brand_cat'] = 'Luxury'
    data.loc[data['brand'].str.contains(premium_brands, na=False, case=False), 'brand_cat'] = 'Premium'
    data.loc[data['brand'].str.contains(mainstream_brands, na=False, case=False), 'brand_cat'] = 'Mainstream'
    data.loc[data['brand'].str.contains(economy_brands, na=False, case=False), 'brand_cat'] = 'Economy'
    data.loc[data['brand'].str.contains(supercar_brands, na=False, case=False), 'brand_cat'] = 'Supercars'

    data['ext_col_cat'] = 'Other'
    data.loc[data['ext_col'].str.contains('Black|Midnight|Onyx|Graphite|Dark|Gray', na=False, case=False), 'ext_col_cat'] = 'Black/Gray'
    data.loc[data['ext_col'].str.contains('White|Pearl|Silver|Platinum|Metallic', na=False, case=False), 'ext_col_cat'] = 'White/Silver'
    data.loc[data['ext_col'].str.contains('Red|Ruby|Burgundy', na=False, case=False), 'ext_col_cat'] = 'Red/Maroon'
    data.loc[data['ext_col'].str.contains('Blue|Navy|Sky', na=False, case=False), 'ext_col_cat'] = 'Blue'
    data.loc[data['ext_col'].str.contains('Green|Forest', na=False, case=False), 'ext_col_cat'] = 'Green'
    data.loc[data['ext_col'].str.contains('Yellow|Gold|Mustard', na=False, case=False), 'ext_col_cat'] = 'Yellow/Gold'
    data.loc[data['ext_col'].str.contains('Brown|Bronze|Tan|Beige', na=False, case=False), 'ext_col_cat'] = 'Brown/Beige'
    data.loc[data['ext_col'].str.contains('Orange|opper', na=False, case=False), 'ext_col_cat'] = 'Orange'
    data.loc[data['ext_col'].str.contains('Purple|Lavender', na=False, case=False), 'ext_col_cat'] = 'Purple'

    data['int_col_cat'] = 'Other'
    data.loc[data['int_col'].str.contains('Black|Charcoal|Dark|Gray', na=False, case=False), 'int_col_cat'] = 'Black/Gray'
    data.loc[data['int_col'].str.contains('White|Ivory|Beige', na=False, case=False), 'int_col_cat'] = 'White/Beige'
    data.loc[data['int_col'].str.contains('Red|Burgundy', na=False, case=False), 'int_col_cat'] = 'Red/Maroon'
    data.loc[data['int_col'].str.contains('Blue|Navy', na=False, case=False), 'int_col_cat'] = 'Blue'
    data.loc[data['int_col'].str.contains('Brown|Tan', na=False, case=False), 'int_col_cat'] = 'Brown'

    damage_reported = data['accident'] == 'At least 1 accident or damage reported'
    data.loc[damage_reported, 'accident'] = 1
    data.loc[~damage_reported, 'accident'] = 0

    data = data.drop(['brand','model', 'model_year', 'engine', 'transmission', 'ext_col', 'int_col', 'clean_title'], axis=1)

    for col in ['horse_power', 'engine_size']:
        mean = data[col].mean()
        data.loc[data[col].isna(), col] = mean

    for col in ['cylinder', 'model_class', 'transmission_type', 'fuel_type', 'brand_cat']:
        mode = data[col].mode().iloc[0]
        data.loc[data[col].isna(), col] = mode

    speed_median = data['transmission_speed'].median()
    data.loc[data['transmission_type'] == 'CVT', 'transmission_speed'] = speed_median
    data.loc[data['transmission_type'] == 'Fixed Gear', 'transmission_speed'] = speed_median

    data['transmission_speed'] = data.groupby(['transmission_type'])['transmission_speed'].transform(lambda x : x.fillna(x.median()))

    # EDA
    data['accident'] = data['accident'].astype("int")

    data['milage_age'] = data['milage'] * data['car_age']
    data['milage_age_ratio'] = data['milage'] / data['car_age']
    data['mean_milage_with_age'] = data.groupby(['car_age'])['milage'].transform('mean')
    data['mean_milage_age_ratio_with_age'] = data.groupby(['car_age'])['milage_age_ratio'].transform('mean')

    assert(data.isna().sum().sum() == 0)
    assert(data.where(data==np.inf).sum().sum() == 0)

    return data