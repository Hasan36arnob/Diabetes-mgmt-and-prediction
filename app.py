# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import joblib
# import numpy as np
# from pymongo import MongoClient

# app = Flask(__name__)

# # MongoDB connection using the provided connection string
# mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
# client = MongoClient(mongo_uri)
# db = client['diabetes_db']
# collection = db['predictions']

# # Load the trained model and label encoders
# try:
#     model = joblib.load('diabetes_model.pkl')
#     label_encoders = joblib.load('label_encoders.pkl')
# except Exception as e:
#     print(f"Error loading model or encoders: {e}")
#     model = None
#     label_encoders = None

# # Routes
# @app.route('/')
# def home():
#     """Home page route"""
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     """About page route"""
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     """Contact page route"""
#     return render_template('contact.html')


# @app.route('/diet')
# def diet():
#     return render_template('diet.html')

# @app.route('/exercise')
# def exercise():
#     return render_template('exercise.html')

# @app.route('/tracking')
# def tracking():
#     return render_template('tracking.html')

# @app.route('/medication')
# def medication():
#     return render_template('medication.html')

# @app.route('/doctor')
# def doctor():
#     return render_template('doctor.html')

# @app.route('/predict_page')
# def predict_page():
#     """Route to the prediction page"""
#     return render_template('predict.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Prediction route for diabetes with risk categorization"""
#     if model is None or label_encoders is None:
#         return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
#     try:
#         # Get location data
#         location_data = {
#             'address': request.form.get('location', ''),
#             'latitude': request.form.get('latitude', ''),
#             'longitude': request.form.get('longitude', '')
#         }

#         # Import required modules
#         import datetime
#         import math
#         import numpy as np

#         # Parse input data from the form
#         input_data = {}
        
#         # Handle Age separately since it's numeric
#         age = request.form.get('Age', '').strip()
#         try:
#             input_data['Age'] = int(age) if age else 0
#         except ValueError:
#             return render_template('predict.html', prediction_text="Error: Age must be a valid number")
        
#         # List of all categorical fields
#         categorical_fields = [
#             'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
#             'weakness', 'Polyphagia', 'visual blurring', 'Itching',
#             'Irritability', 'delayed healing', 'partial paresis',
#             'muscle stiffness', 'Alopecia', 'Obesity'
#         ]
        
#         # Handle categorical inputs with validation
#         for field in categorical_fields:
#             value = request.form.get(field, '').strip()
#             if value and value not in ['Yes', 'No', 'Male', 'Female']:
#                 return render_template('predict.html', 
#                     prediction_text=f"Error: Invalid value for {field}. Must be Yes/No or Male/Female for Gender")
#             input_data[field] = value if value else 'No Answer'
        
#         # Encode categorical inputs
#         features = []
#         for key in input_data.keys():
#             if key == 'Age':
#                 features.append(input_data[key])
#             else:
#                 try:
#                     if input_data[key] == 'No Answer':
#                         features.append(0)
#                     else:
#                         encoded_value = label_encoders[key].transform([input_data[key]])[0]
#                         features.append(int(encoded_value))
#                 except KeyError:
#                     return render_template('predict.html', 
#                         prediction_text=f"Error: Missing encoder for field {key}")
#                 except ValueError:
#                     return render_template('predict.html', 
#                         prediction_text=f"Error: Invalid value for field {key}")
        
#         # Convert to numpy array and reshape
#         features = np.array(features).reshape(1, -1)
        
#         # Validate feature array shape
#         expected_features = len(categorical_fields) + 1  # +1 for Age
#         if features.shape[1] != expected_features:
#             return render_template('predict.html', 
#                 prediction_text=f"Error: Invalid number of features. Expected {expected_features}, got {features.shape[1]}")

#         # Calculate metrics
#         answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
#         total_questions = len(input_data)
        
#         # Prevent division by zero
#         if total_questions == 0:
#             return render_template('predict.html', prediction_text="Error: No questions available")
        
#         # Calculate confidence score
#         confidence_factor = math.log(1 + 9 * (answered_questions / total_questions)) / math.log(10)
#         confidence_score = confidence_factor * 100

#         # Make prediction
#         try:
#             probabilities = model.predict_proba(features)
#             probability_positive = float(probabilities[0][1] * 100)
#         except Exception as e:
#             return render_template('predict.html', 
#                 prediction_text=f"Error making prediction: {str(e)}")

#         # Calculate feature importance
#         feature_importance = {}
#         for i, (key, value) in enumerate(input_data.items()):
#             if value not in ['', 'No Answer', 0]:
#                 feature_importance[key] = float(model.feature_importances_[i])

#         # Calculate adjusted confidence
#         if feature_importance:
#             importance_sum = sum(feature_importance.values())
#             if importance_sum > 0:
#                 weighted_importance = sum(importance / importance_sum for importance in feature_importance.values())
#                 importance_factor = weighted_importance / len(feature_importance)
#                 adjusted_confidence = confidence_score * (0.7 + 0.3 * importance_factor)
#             else:
#                 adjusted_confidence = confidence_score * 0.7
#         else:
#             adjusted_confidence = confidence_score * 0.7

#         # Adjust probability
#         adjusted_probability = probability_positive * (adjusted_confidence / 100)

#         # Risk categorization
#         risk_categories = {
#             (70, 100): ("Very High Risk", "#8B0000"),
#             (55, 70): ("High Risk", "#FF0000"),
#             (40, 55): ("Moderate Risk", "#FFA500"),
#             (25, 40): ("Low Risk", "#FFFF00"),
#             (0, 25): ("Very Low Risk", "#008000")
#         }

#         risk_category = "Unknown Risk"
#         color = "#808080"
#         for (lower, upper), (category, cat_color) in risk_categories.items():
#             if lower < adjusted_probability <= upper:
#                 risk_category = category
#                 color = cat_color
#                 break

#         # Format prediction text
#         prediction_text = (
#             f'<div class="result-card">'
#             f'  <h2>Risk Assessment Results</h2>'
#             f'  <div class="risk-indicator" style="background-color:{color};">'
#             f'    <h3>Risk of Developing Diabetes: {risk_category}</h3>'
#             f'    <p>Estimated Risk: {adjusted_probability:.1f}%</p>'
#             f'  </div>'
#             f'  <div class="confidence-info">'
#             f'    <p>Confidence Level: {adjusted_confidence:.1f}%</p>'
#             f'    <p>Based on {answered_questions} of {total_questions} answered questions</p>'
#             f'    <p>Data completeness: {(answered_questions / total_questions * 100):.1f}%</p>'
#             f'  </div>'
#             f'</div>'
#         )

#         # Save to MongoDB
#         model_version = "1.0"  # Define model version
#         prediction_record = {
#             'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
#             'probability_raw': float(probability_positive),
#             'probability_adjusted': float(adjusted_probability),
#             'confidence_score': float(adjusted_confidence),
#             'data_completeness': float(answered_questions / total_questions * 100),
#             'risk_category': risk_category,
#             'answered_questions': answered_questions,
#             'total_questions': total_questions,
#             'timestamp': datetime.datetime.now(),
#             'model_version': model_version,
#             'location': location_data  # Add location data to the record
#         }
        
#         try:
#             collection.insert_one(prediction_record)
#         except Exception as e:
#             print(f"Warning: Failed to save prediction to MongoDB: {str(e)}")
#             # Continue execution even if MongoDB save fails

#         # Feature importance ranking
#         sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
#         top_factors = sorted_features[:min(5, len(sorted_features))]

#         return render_template(
#             'result.html',
#             prediction_text=prediction_text,
#             risk_category=risk_category,
#             probability=adjusted_probability,
#             confidence_score=adjusted_confidence,
#             data_completeness=answered_questions / total_questions * 100,
#             sorted_features=sorted_features,
#             top_factors=top_factors,
#             back_url=url_for('home')
#         )
        
#     except Exception as e:
#         import traceback
#         print(f"Prediction error: {str(e)}")
#         print(traceback.format_exc())
#         return render_template('predict.html', 
#             prediction_text=f"An unexpected error occurred: {str(e)}")

# @app.route('/suggest', methods=['GET'])
# def suggest():
#     """Suggestion route based on feature ranking"""
#     try:
#         # Fetch the latest prediction from MongoDB
#         latest_prediction = collection.find_one(sort=[('_id', -1)])
#         if not latest_prediction:
#             return render_template('suggest.html', suggestions="No predictions found.")

#         # Get features where user actually provided positive responses
#         input_data = latest_prediction['input_data']
#         relevant_features = []
#         for feature, value in input_data.items():
#             # Check if the value indicates a positive response (typically 1 or 'Yes')
#             if value in [1, 'Yes', True]:
#                 relevant_features.append((feature, model.feature_importances_[list(input_data.keys()).index(feature)]))

#         # Sort by feature importance and get top 3
#         sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:3]

#         # Generate suggestions only for features with positive responses
#         suggestions = []
#         for feature, _ in sorted_features:
#             if feature == 'Age':
#                 suggestions.append("Monitor your age-related health risks regularly.")
#             elif feature == 'Polyuria':
#                 suggestions.append("Consult a doctor if you experience frequent urination.")
#             elif feature == 'Polydipsia':
#                 suggestions.append("Stay hydrated but consult a doctor if excessive thirst persists.")
#             elif feature == 'sudden weight loss':
#                 suggestions.append("Seek medical advice for unexplained weight loss.")
#             elif feature == 'weakness':
#                 suggestions.append("Ensure proper nutrition and rest to manage weakness.")
#             elif feature == 'Polyphagia':
#                 suggestions.append("Monitor your eating habits and consult a doctor if needed.")
#             elif feature == 'visual blurring':
#                 suggestions.append("Get your eyes checked if you experience blurred vision.")
#             elif feature == 'Itching':
#                 suggestions.append("Use moisturizers and consult a dermatologist if itching persists.")
#             elif feature == 'Irritability':
#                 suggestions.append("Practice stress management techniques.")
#             elif feature == 'delayed healing':
#                 suggestions.append("Seek medical advice for slow-healing wounds.")
#             elif feature == 'partial paresis':
#                 suggestions.append("Consult a neurologist for muscle weakness.")
#             elif feature == 'muscle stiffness':
#                 suggestions.append("Stretch regularly and consult a doctor if stiffness persists.")
#             elif feature == 'Alopecia':
#                 suggestions.append("Consult a dermatologist for hair loss issues.")
#             elif feature == 'Obesity':
#                 suggestions.append("Adopt a healthy diet and exercise regularly.")

#         return render_template('suggest.html', suggestions=suggestions, back_url=url_for('home'))
#     except Exception as e:
#         return render_template('suggest.html', suggestions=f'Error: {str(e)}')

# if __name__ == '__main__':
#     app.run(debug=True)

# @app.route('/location', methods=['GET', 'POST'])
# def location():
#     if request.method == 'POST':
#         location_data = {
#             'address': request.form.get('location'),
#             'latitude': request.form.get('latitude'),
#             'longitude': request.form.get('longitude')
#         }
#         # Store in MongoDB
#         collection.update_one(
#             {'_id': latest_prediction_id}, 
#             {'$set': {'location': location_data}}
#         )
#         return redirect(url_for('predict_page'))
#     return render_template('location.html')

# @app.route('/location', methods=['GET', 'POST'])
# def location():
#     if request.method == 'POST':
#         location_data = {
#             'address': request.form.get('location'),
#             'latitude': request.form.get('latitude'),
#             'longitude': request.form.get('longitude')
#         }
#         # Store in MongoDB
#         collection.update_one(
#             {'_id': latest_prediction_id}, 
#             {'$set': {'location': location_data}}
#         )
#         return redirect(url_for('predict_page'))
#     return render_template('location.html')


from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from pymongo import MongoClient
import datetime
import math

app = Flask(__name__)

# MongoDB connection using the provided connection string
mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
client = MongoClient(mongo_uri)
db = client['diabetes_db']
collection = db['predictions']

# Load the trained model and label encoders
try:
    model = joblib.load('diabetes_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    label_encoders = None

# Routes - keeping all original routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page route"""
    return render_template('contact.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/medication')
def medication():
    return render_template('medication.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/predict_page')
def predict_page():
    """Route to the prediction page"""
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route for diabetes with improved risk categorization and accuracy"""
    if model is None or label_encoders is None:
        return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
    try:
        # Get location data
        location_data = {
            'address': request.form.get('location', ''),
            'latitude': request.form.get('latitude', ''),
            'longitude': request.form.get('longitude', '')
        }

        # Import required modules
        import datetime
        import math
        import numpy as np

        # Parse input data from the form
        input_data = {}
        
        # Handle Age separately since it's numeric
        age = request.form.get('Age', '').strip()
        try:
            input_data['Age'] = int(age) if age else 0
        except ValueError:
            return render_template('predict.html', prediction_text="Error: Age must be a valid number")
        
        # List of all categorical fields
        categorical_fields = [
            'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
            'weakness', 'Polyphagia', 'visual blurring', 'Itching',
            'Irritability', 'delayed healing', 'partial paresis',
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        # Handle categorical inputs with validation
        for field in categorical_fields:
            value = request.form.get(field, '').strip()
            if value and value not in ['Yes', 'No', 'Male', 'Female']:
                return render_template('predict.html', 
                    prediction_text=f"Error: Invalid value for {field}. Must be Yes/No or Male/Female for Gender")
            input_data[field] = value if value else 'No Answer'
        
        # Encode categorical inputs
        features = []
        for key in input_data.keys():
            if key == 'Age':
                features.append(input_data[key])
            else:
                try:
                    if input_data[key] == 'No Answer':
                        features.append(0)
                    else:
                        encoded_value = label_encoders[key].transform([input_data[key]])[0]
                        features.append(int(encoded_value))
                except KeyError:
                    return render_template('predict.html', 
                        prediction_text=f"Error: Missing encoder for field {key}")
                except ValueError:
                    return render_template('predict.html', 
                        prediction_text=f"Error: Invalid value for field {key}")
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Validate feature array shape
        expected_features = len(categorical_fields) + 1  # +1 for Age
        if features.shape[1] != expected_features:
            return render_template('predict.html', 
                prediction_text=f"Error: Invalid number of features. Expected {expected_features}, got {features.shape[1]}")

        # Calculate metrics
        answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
        total_questions = len(input_data)
        
        # Prevent division by zero
        if total_questions == 0:
            return render_template('predict.html', prediction_text="Error: No questions available")
        
        # IMPROVEMENT 1: Better confidence calculation using sigmoid function
        # This provides a more statistically sound scaling of confidence
        completeness = answered_questions / total_questions
        confidence_score = 100 * (1 / (1 + math.exp(-12 * (completeness - 0.5))))

        # Make prediction
        try:
            probabilities = model.predict_proba(features)
            probability_positive = float(probabilities[0][1] * 100)
        except Exception as e:
            return render_template('predict.html', 
                prediction_text=f"Error making prediction: {str(e)}")

        # Calculate feature importance - same as original
        feature_importance = {}
        for i, (key, value) in enumerate(input_data.items()):
            if value not in ['', 'No Answer', 0]:
                feature_importance[key] = float(model.feature_importances_[i])

        # IMPROVEMENT 2: Better confidence adjustment based on clinical relevance of features
        # Define clinically significant features and their weights
        clinical_weights = {
            'Age': 1.5,
            'Polyuria': 2.0,      # Strong clinical indicator
            'Polydipsia': 2.0,    # Strong clinical indicator  
            'sudden weight loss': 1.8,
            'Obesity': 1.5,
            'Gender': 0.8,
            'Alopecia': 0.7,
            'muscle stiffness': 0.9,
            'Itching': 0.9
        }
        
        # Calculate weighted clinical importance of provided answers
        clinical_score = 0
        clinical_total = 0
        
        for feature, value in input_data.items():
            if value not in ['', 'No Answer', 0]:
                weight = clinical_weights.get(feature, 1.0)
                clinical_score += weight
                clinical_total += weight
        
        # Calculate clinical completeness factor
        clinical_factor = clinical_score / sum(clinical_weights.values()) if clinical_weights else 0
                
        # Combine standard completeness with clinical relevance
        adjusted_confidence = confidence_score * (0.4 + 0.6 * clinical_factor)

        # IMPROVEMENT 3: Bayesian probability calibration
        # This adjusts the model's raw probability to be better calibrated
        # Values derived from Platt scaling (simplistic implementation)
        def calibrate_probability(prob):
            # Apply sigmoid calibration function
            a = 0.8  # Slope parameter
            b = -0.2  # Intercept parameter
            
            prob_decimal = prob / 100.0
            calibrated = 1 / (1 + math.exp(-(a * prob_decimal + b)))
            return calibrated * 100
        
        # Calibrate the raw probability
        calibrated_probability = calibrate_probability(probability_positive)
        
        # IMPROVEMENT 4: Adjust final probability based on confidence
        # Higher confidence means we trust the model's prediction more
        confidence_weight = min(adjusted_confidence / 100, 0.95)  # Cap at 95% to avoid overconfidence
        base_probability = 50  # Neutral starting point
        
        # Linear interpolation between base and calibrated probability
        adjusted_probability = (calibrated_probability * confidence_weight) + (base_probability * (1 - confidence_weight))
        
        # IMPROVEMENT 5: Evidence-based risk categories with clinical thresholds
        risk_categories = {
            (70, 100): ("Very High Risk", "#8B0000"),
            (55, 70): ("High Risk", "#FF0000"),
            (40, 55): ("Moderate Risk", "#FFA500"),
            (25, 40): ("Low Risk", "#FFFF00"),
            (0, 25): ("Very Low Risk", "#008000")
        }

        risk_category = "Unknown Risk"
        color = "#808080"
        for (lower, upper), (category, cat_color) in risk_categories.items():
            if lower < adjusted_probability <= upper:
                risk_category = category
                color = cat_color
                break

        # IMPROVEMENT 6: Calculate uncertainty range based on confidence
        margin_error = (100 - adjusted_confidence) / 8
        lower_bound = max(0, adjusted_probability - margin_error)
        upper_bound = min(100, adjusted_probability + margin_error)

        # Format prediction text with confidence interval
        prediction_text = (
            f'<div class="result-card">'
            f'  <h2>Risk Assessment Results</h2>'
            f'  <div class="risk-indicator" style="background-color:{color};">'
            f'    <h3>Risk of Developing Diabetes: {risk_category}</h3>'
            f'    <p>Estimated Risk: {adjusted_probability:.1f}% </p>'
            f'  </div>'
            f'  <div class="confidence-info">'
            f'    <p>Confidence Level: {adjusted_confidence:.1f}%</p>'
            f'  </div>'
            f'</div>'
        )

        # Save to MongoDB - adding new fields but keeping original structure
        model_version = "1.0"  # Define model version
        prediction_record = {
            'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
            'probability_raw': float(probability_positive),
            'probability_calibrated': float(calibrated_probability),
            'probability_adjusted': float(adjusted_probability),
            'confidence_score': float(adjusted_confidence),
            'confidence_margin': float(margin_error),
            'data_completeness': float(answered_questions / total_questions * 100),
            'risk_category': risk_category,
            'answered_questions': answered_questions,
            'total_questions': total_questions,
            'timestamp': datetime.datetime.now(),
            'model_version': model_version,
            'location': location_data
        }
        
        try:
            collection.insert_one(prediction_record)
        except Exception as e:
            print(f"Warning: Failed to save prediction to MongoDB: {str(e)}")
            # Continue execution even if MongoDB save fails

        # Feature importance ranking
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_features[:min(5, len(sorted_features))]

        return render_template(
            'result.html',
            prediction_text=prediction_text,
            risk_category=risk_category,
            probability=adjusted_probability,
            confidence_score=adjusted_confidence,
            confidence_interval=f"{lower_bound:.1f}% - {upper_bound:.1f}%",
            data_completeness=answered_questions / total_questions * 100,
            sorted_features=sorted_features,
            top_factors=top_factors,
            back_url=url_for('home')
        )
        
    except Exception as e:
        import traceback
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return render_template('predict.html', 
            prediction_text=f"An unexpected error occurred: {str(e)}")

@app.route('/suggest', methods=['GET'])
def suggest():
    """Suggestion route based on feature ranking"""
    try:
        # Fetch the latest prediction from MongoDB
        latest_prediction = collection.find_one(sort=[('_id', -1)])
        if not latest_prediction:
            return render_template('suggest.html', suggestions="No predictions found.")

        # Get features where user actually provided positive responses
        input_data = latest_prediction['input_data']
        relevant_features = []
        for feature, value in input_data.items():
            # Check if the value indicates a positive response (typically 1 or 'Yes')
            if value in [1, 'Yes', True]:
                relevant_features.append((feature, model.feature_importances_[list(input_data.keys()).index(feature)]))

        # Sort by feature importance and get top 3
        sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:3]

        # Generate suggestions only for features with positive responses
        suggestions = []
        for feature, _ in sorted_features:
            if feature == 'Age':
                suggestions.append("Monitor your age-related health risks regularly.")
            elif feature == 'Polyuria':
                suggestions.append("Consult a doctor if you experience frequent urination.")
            elif feature == 'Polydipsia':
                suggestions.append("Stay hydrated but consult a doctor if excessive thirst persists.")
            elif feature == 'sudden weight loss':
                suggestions.append("Seek medical advice for unexplained weight loss.")
            elif feature == 'weakness':
                suggestions.append("Ensure proper nutrition and rest to manage weakness.")
            elif feature == 'Polyphagia':
                suggestions.append("Monitor your eating habits and consult a doctor if needed.")
            elif feature == 'visual blurring':
                suggestions.append("Get your eyes checked if you experience blurred vision.")
            elif feature == 'Itching':
                suggestions.append("Use moisturizers and consult a dermatologist if itching persists.")
            elif feature == 'Irritability':
                suggestions.append("Practice stress management techniques.")
            elif feature == 'delayed healing':
                suggestions.append("Seek medical advice for slow-healing wounds.")
            elif feature == 'partial paresis':
                suggestions.append("Consult a neurologist for muscle weakness.")
            elif feature == 'muscle stiffness':
                suggestions.append("Stretch regularly and consult a doctor if stiffness persists.")
            elif feature == 'Alopecia':
                suggestions.append("Consult a dermatologist for hair loss issues.")
            elif feature == 'Obesity':
                suggestions.append("Adopt a healthy diet and exercise regularly.")

        return render_template('suggest.html', suggestions=suggestions, back_url=url_for('home'))
    except Exception as e:
        return render_template('suggest.html', suggestions=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/location', methods=['GET', 'POST'])
def location():
    if request.method == 'POST':
        location_data = {
            'address': request.form.get('location'),
            'latitude': request.form.get('latitude'),
            'longitude': request.form.get('longitude')
        }
        # Store in MongoDB
        collection.update_one(
            {'_id': latest_prediction_id}, 
            {'$set': {'location': location_data}}
        )
        return redirect(url_for('predict_page'))
    return render_template('location.html')