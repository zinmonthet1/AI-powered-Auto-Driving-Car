import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__)

def car_data_transformer(df_input):
    X_copy = df_input.copy()
    if 'speed_limit_kmh' in X_copy.columns:
        X_copy['speed_limit_mps'] = X_copy['speed_limit_kmh'] / 3.6
        X_copy = X_copy.drop(columns=['speed_limit_kmh'])
    road_mapping = {'dry': 3, 'wet': 2, 'icy': 1}
    if 'road_surface_condition' in X_copy.columns:
        X_copy['road_surface_condition'] = X_copy['road_surface_condition'].map(road_mapping).fillna(3)
    return X_copy

model = joblib.load('autoDriving_pipeline.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/vehicles')
def vehicles():
    return render_template('vehicles.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    decision = None
    if request.method == 'POST':
        # Collect all 18 features from form
        data = {
            'obstacle_distance_m': float(request.form.get('obstacle_distance_m', 50)),
            'relative_speed_mps': float(request.form.get('relative_speed_mps', 0)),
            'num_obstacles': int(request.form.get('num_obstacles', 1)),
            'lane_offset_m': float(request.form.get('lane_offset_m', 0)),
            'traffic_density_veh_per_km': float(request.form.get('traffic_density_veh_per_km', 20)),
            'risk_probability': float(request.form.get('risk_probability', 0.1)),
            'road_curvature_1pm': float(request.form.get('road_curvature_1pm', 0)),
            'road_width_m': float(request.form.get('road_width_m', 3.5)),
            'speed_limit_kmh': float(request.form.get('speed_limit_kmh', 60)),
            'ego_speed_mps': float(request.form.get('ego_speed_mps', 15)),
            'ego_acceleration_mps2': float(request.form.get('ego_acceleration_mps2', 0)),
            'steering_angle_deg': float(request.form.get('steering_angle_deg', 0)),
            'yaw_rate_rads': float(request.form.get('yaw_rate_rads', 0)),
            'throttle_position': float(request.form.get('throttle_position', 0.2)),
            'brake_pressure': float(request.form.get('brake_pressure', 0)),
            'visibility_range_m': float(request.form.get('visibility_range_m', 200)),
            'weather_condition': request.form.get('weather_condition', 'clear'),
            'road_surface_condition': request.form.get('road_surface_condition', 'dry')
        }
        input_df = pd.DataFrame([data])
        prediction_idx = model.predict(input_df)[0]
        decision = model.target_names[prediction_idx]
        
    return render_template('predict.html', decision=decision)

if __name__ == '__main__':
    app.run(debug=True)
    
#/opt/anaconda3/envs/streamlit_env/bin/python flaskapp.py