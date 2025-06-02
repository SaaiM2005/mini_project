from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Attention, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import os
import logging
from geopy.distance import geodesic
import math
import tensorflow as tf
from datetime import timedelta

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

AVERAGE_OSPREY_SPEED_MPS = 13.41  # ~30 mph in meters per second

def convert_to_float(data):
    return [{"lat": float(point["lat"]), "lon": float(point["lon"]), "timestamp": point["timestamp"]} for point in data]

def calculate_speed_and_bearing(df):
    speeds = [0]
    bearings = [0]
    for i in range(1, len(df)):
        coord1 = (df.iloc[i-1]['location-lat'], df.iloc[i-1]['location-long'])
        coord2 = (df.iloc[i]['location-lat'], df.iloc[i]['location-long'])
        dist = geodesic(coord1, coord2).meters
        time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 3600.0
        speed = dist / time_diff if time_diff > 0 else 0
        bearing = math.atan2(df.iloc[i]['location-long'] - df.iloc[i-1]['location-long'],
                             df.iloc[i]['location-lat'] - df.iloc[i-1]['location-lat'])
        speeds.append(speed)
        bearings.append(bearing)
    df['speed'] = speeds
    df['bearing'] = bearings
    return df

@app.route('/get_bird_ids', methods=['GET'])
def get_bird_ids():
    csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_tracking_data.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'cleaned_tracking_data.csv not found on the server'}), 500

    df = pd.read_csv(csv_path)
    bird_ids = df['individual-local-identifier'].unique().tolist()

    return jsonify({'bird_ids': bird_ids})


@app.route('/generate_map', methods=['POST'])
def generate_map():
    bird_id = request.json.get('bird_id')
    if not bird_id:
        return jsonify({'error': 'Bird ID not provided'}), 400

    logging.info(f"Received request for bird_id: {bird_id}")

    csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_tracking_data.csv')
    if not os.path.exists(csv_path):
        logging.error("cleaned_tracking_data.csv not found!")
        return jsonify({'error': 'cleaned_tracking_data.csv not found on the server'}), 500

    df = pd.read_csv(csv_path)
    df = df[df['individual-local-identifier'] == bird_id]
    if df.empty:
        logging.warning(f"No data found for bird: {bird_id}")
        return jsonify({'error': f"No data found for bird {bird_id}"}), 404

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = calculate_speed_and_bearing(df)

    features = ['location-long', 'location-lat', 'speed', 'bearing']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    seq_length = 10
    X, y, timestamps = [], [], []
    for i in range(len(df) - seq_length):
        X.append(df[features].iloc[i:i+seq_length].values)
        y.append(df[features].iloc[i+seq_length][['location-long', 'location-lat']].values)
        timestamps.append(df['timestamp'].iloc[i+seq_length])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return jsonify({'error': 'Not enough data to train the model'}), 400

    input_layer = Input(shape=(seq_length, len(features)))
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    attention = Attention()([bi_lstm, bi_lstm])
    context_vector = GlobalAveragePooling1D()(attention)
    output = Dense(2)(context_vector)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)

    predicted = model.predict(X)
    predicted_latlon = scaler.inverse_transform(np.hstack([predicted, np.zeros((predicted.shape[0], 2))]))[:, :2]
    actual_latlon = scaler.inverse_transform(np.hstack([y, np.zeros((y.shape[0], 2))]))[:, :2]

    # Predict 7 days ahead based on constant speed
    total_steps = 24 * 7
    last_sequence = X[-1]
    last_timestamp = timestamps[-1]
    future_predictions = []

    for _ in range(total_steps):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features)))
        extended = np.hstack([next_pred[0], [0, 0]])
        rescaled = scaler.inverse_transform(extended.reshape(1, -1))[:, :2][0]

        # Estimate distance from previous point to calculate time
        if future_predictions:
            prev = (future_predictions[-1]["lat"], future_predictions[-1]["lon"])
        else:
            prev = (actual_latlon[-1][1], actual_latlon[-1][0])
        curr = (rescaled[1], rescaled[0])
        dist_m = geodesic(prev, curr).meters
        seconds = dist_m / AVERAGE_OSPREY_SPEED_MPS
        last_timestamp += timedelta(seconds=seconds)

        future_predictions.append({
            "lat": rescaled[1],
            "lon": rescaled[0],
            "timestamp": last_timestamp.strftime('%b %d, %Y - %I:%M %p')
        })

        extended_scaled = scaler.transform(np.hstack([next_pred, [[0, 0]]]))[0]
        last_sequence = np.vstack([last_sequence[1:], extended_scaled])

    actual = [{"lat": float(lat), "lon": float(lon)} for lon, lat in actual_latlon]
    predicted = [{"lat": float(lat), "lon": float(lon)} for lon, lat in predicted_latlon]

    return jsonify({
        "actual": actual,
        "predicted": predicted,
        "future": future_predictions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
