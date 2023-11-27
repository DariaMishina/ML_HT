from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from io import BytesIO
import re

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def load_model():
    with open('poly_scaler_model.pkl', 'rb') as file:
        poly, scaler, model = joblib.load(file)
    return poly, scaler, model

poly, scaler, model = load_model()

def extract_numeric(value):
    numeric_part = re.search(r'\d+\.\d+|\d+', str(value))
    return float(numeric_part.group()) if numeric_part else None


def preprocess_data(data):
    new_data = pd.DataFrame(data.dict(), index=[0])
    new_data['mileage'] = new_data['mileage'].apply(extract_numeric)
    new_data['engine'] = new_data['engine'].apply(extract_numeric)
    new_data['max_power'] = new_data['max_power'].apply(extract_numeric)
    new_data.drop(['torque'], axis=1, inplace=True)
    new_data[['mileage', 'engine', 'max_power', 'seats']] = new_data[
        ['mileage', 'engine', 'max_power', 'seats']].fillna(0)
    new_data['engine'] = new_data['engine'].astype('int')
    new_data['seats'] = new_data['seats'].astype('int')

    new_data = new_data.select_dtypes(exclude=['object'])
    new_data = new_data.drop('selling_price', axis=1)

    new_data_pl = poly.transform(new_data)
    new_data = pd.DataFrame(new_data_pl, index=new_data.index, columns=poly.get_feature_names_out(new_data.columns))

    new_data_scaled = scaler.transform(new_data)
    new_data_df = pd.DataFrame(new_data_scaled, index=new_data.index, columns=new_data.columns)
    return new_data_df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    new_data_df = preprocess_data(item)
    prediction = model.predict(new_data_df)
    return round(float(np.expm1(prediction)))


@app.post("/predict_items")
def predict_items(file: UploadFile):
    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer, index_col=0)
    buffer.close()
    file.close()

    predictions = []
    for _, row in df.iterrows():
        # cоздаем объект Item из строки датафрейма
        item = Item(
            name=row.get('name', ''),
            year=row.get('year', 0),
            selling_price=row.get('selling_price', 0),
            km_driven=row.get('km_driven', 0),
            fuel=row.get('fuel', ''),
            seller_type=row.get('seller_type', ''),
            transmission=row.get('transmission', ''),
            owner=row.get('owner', ''),
            mileage=row.get('mileage', ''),
            engine=row.get('engine', ''),
            max_power=row.get('max_power', ''),
            torque=row.get('torque', ''),
            seats=row.get('seats', 0.0)
        )

        new_data_df = preprocess_data(item)
        prediction = model.predict(new_data_df)
        predictions.append(round(float(np.expm1(prediction))))

    df['predictions'] = predictions

    output_file = "predictions_output.csv"
    df.to_csv(output_file, index=False)
    return output_file