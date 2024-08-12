import os
import json
import requests
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient
import logging
from logging.handlers import RotatingFileHandler
from google import generativeai as genai
from datetime import datetime, timedelta
import pandas as pd
import google_auth_oauthlib.flow
import googleapiclient.discovery
import uuid
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.io as pio
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
from grpc._channel import _Rendezvous as ResourceExhausted
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image
import markdown2
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')
login_manager = LoginManager()
login_manager.init_app(app)
transactions = []
tips_data = {
    'date': None,
    'tips': {}
}
employees = []
tasks = []
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
current_tasks = tasks if tasks else []
# Load credentials from file
with open('credentials.json') as f:
    credentials = json.load(f)

GOOGLE_CLIENT_ID = credentials['client_id']
GOOGLE_CLIENT_SECRET = credentials['client_secret']
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# In-memory storage
profiles_storage = {}
inventory_storage = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class User(UserMixin):
    def __init__(self, id_):
        self.id = id_

    @staticmethod
    def get(user_id):
        return User(user_id)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    try:
        google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
        authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    except requests.RequestException as e:
        logger.error(f"Error fetching Google provider configuration: {e}")
        return f"Error fetching Google provider configuration: {e}", 500

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"]
    )
    return redirect(request_uri)

@app.route('/login/callback')
def callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for('login'))

    try:
        google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
        token_endpoint = google_provider_cfg["token_endpoint"]
        token_url, headers, body = client.prepare_token_request(
            token_endpoint,
            authorization_response=request.url,
            redirect_url=request.base_url,
            code=code
        )
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
        )
        client.parse_request_body_response(json.dumps(token_response.json()))

        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        uri, headers, body = client.add_token(userinfo_endpoint)
        userinfo_response = requests.get(uri, headers=headers, data=body)
        user_info = userinfo_response.json()

        user = User(user_info["sub"])
        login_user(user)
        return redirect(url_for("profile"))
    except requests.RequestException as e:
        logger.error(f"Error during callback processing: {e}")
        return f"Error during callback processing: {e}", 500

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            profile_data = {
                "company": request.form.get('company', ''),
                "company_age": request.form.get('company_age', ''),
                "email": request.form.get('email', ''),
                "phone": request.form.get('phone', ''),
                "industry": request.form.getlist('industry'),
                "description": request.form.get('description', ''),
                "goals": request.form.getlist('goals'),
                "goals_other": request.form.get('goals_other', ''),
                "challenges": request.form.getlist('challenges'),
                "challenges_other": request.form.get('challenges_other', ''),
                "employees": request.form.get('employees', ''),
                "revenue": request.form.get('revenue', ''),
                "target_market": request.form.get('target_market', ''),
                "marketing_channels": request.form.getlist('marketing_channels'),
                "marketing_channels_other": request.form.get('marketing_channels_other', ''),
                "technology_stack": request.form.get('technology_stack', ''),
                "kpis": request.form.getlist('kpis'),
                "kpi_values": request.form.get('kpi_values', '{}'),
                "contact": request.form.get('contact', '')
            }
            # Convert kpi_values from JSON string to dictionary
            try:
                profile_data["kpi_values"] = json.loads(profile_data["kpi_values"])
            except json.JSONDecodeError:
                profile_data["kpi_values"] = {}

            profiles_storage[current_user.id] = profile_data
            flash("Profile saved successfully", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            flash("An error occurred while saving your profile. Please try again.", "danger")
            return redirect(url_for("profile"))

    profile = profiles_storage.get(current_user.id, {})
    return render_template('profile.html', profile=profile)

@app.route('/saved-profile', methods=['GET', 'POST'])
@login_required
def saved_profile():
    user_id = current_user.id
    if request.method == 'POST':
        try:
            profile_data = {
                "company": request.form.get('company', ''),
                "company_age": request.form.get('company_age', ''),
                "email": request.form.get('email', ''),
                "phone": request.form.get('phone', ''),
                "industry": request.form.getlist('industry'),
                "description": request.form.get('description', ''),
                "goals": request.form.getlist('goals'),
                "goals_other": request.form.get('goals_other', ''),
                "challenges": request.form.getlist('challenges'),
                "challenges_other": request.form.get('challenges_other', ''),
                "employees": request.form.get('employees', ''),
                "revenue": request.form.get('revenue', ''),
                "target_market": request.form.get('target_market', ''),
                "marketing_channels": request.form.getlist('marketing_channels'),
                "marketing_channels_other": request.form.get('marketing_channels_other', ''),
                "technology_stack": request.form.get('technology_stack', ''),
                "kpis": request.form.getlist('kpis'),
                "kpi_values": request.form.get('kpi_values', '{}'),
                "contact": request.form.get('contact', '')
            }
            # Convert kpi_values from JSON string to dictionary
            try:
                profile_data["kpi_values"] = json.loads(profile_data["kpi_values"])
            except json.JSONDecodeError:
                profile_data["kpi_values"] = {}

            profiles_storage[user_id] = profile_data
            flash("Profile updated successfully!", "success")
            return redirect(url_for("saved_profile"))
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            flash("An error occurred while saving your profile. Please try again.", "danger")
            return redirect(url_for("saved_profile"))

    profile = profiles_storage.get(user_id, {})
    return render_template('saved_profile.html', profile=profile)

@app.route('/dashboard')
@login_required
def dashboard():
    profile = profiles_storage.get(current_user.id, {})
    business_name = profile.get("company", "Business Name")
    return render_template('dashboard.html', business_name=business_name)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
    return "Internal Server Error", 500

# Configure GenAI with API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Error: Please set the GEMINI_API_KEY environment variable with your GenAI API key.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-1.5-pro')

@app.route('/ask_gemini', methods=['GET'])
def ask_gemini_page():
    return render_template('ask_gemini.html')

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    user_input = request.form.get('prompt')
    image_file = request.files.get('image')

    if not user_input:
        return jsonify({'error': 'No prompt provided'}), 400

    response_data = {}

    try:
        # Process the prompt and optionally an image
        if image_file:
            image = PIL.Image.open(image_file)
            response = model.generate_content([image, user_input])
        else:
            response = model.generate_content(user_input)

        # Check if the response contains generated content
        if response.parts:
            generated_text = "".join(part.text for part in response.parts)
            html_text = markdown2.markdown(generated_text)
            response_data['response'] = html_text
        else:
            # Handle cases where no valid content was generated
            safety_ratings = [str(candidate.safety_ratings) for candidate in response.candidates]
            return jsonify({
                'error': 'No valid content was generated. Please check the safety ratings.',
                'safety_ratings': safety_ratings
            }), 200

        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error generating response: {str(e)}")
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/tips_of_the_day', methods=['GET'])
@login_required
def tips_of_the_day():
    global tips_data
    current_date = datetime.now().date()
    user_id = current_user.id

    if tips_data['date'] != current_date or user_id not in tips_data['tips']:
        profile_data = profiles_storage.get(user_id, {})
        if not profile_data:
            return jsonify({'error': 'Profile data not found'}), 404
        
        try:
            tips = generate_tips(profile_data, current_date)
            tips_data['tips'][user_id] = tips
            tips_data['date'] = current_date
        except Exception as e:
            logger.error(f"Error generating tips: {e}", exc_info=True)
            return jsonify({'error': 'An error occurred while generating tips.'}), 500

    return render_template('tips_of_the_day.html', tips=tips_data['tips'][user_id])

def generate_tips(profile_data, current_date):
    industry = profile_data.get('industry', [''])[0]
    description = profile_data.get('description', 'their business')
    company_age = profile_data.get('company_age', 'an established')
    revenue = profile_data.get('revenue', 'undisclosed')
    goals = ", ".join(profile_data.get('goals', ['unspecified goals']))
    challenges = ", ".join(profile_data.get('challenges', ['unspecified challenges']))

    prompts = [
        f"Generate a business tip for a {industry} business described as {description}.",
        f"How can a {company_age} {industry} business increase its revenue from {revenue}?",
        f"What are some tips for managing employees in a {industry} business facing challenges like {challenges}?",
        f"Suggest marketing strategies for a {industry} business with goals such as {goals}.",
        f"How can a {industry} business improve customer satisfaction with a business described as {description}?"
    ]

    # Use the current date to rotate prompts
    prompt_index = current_date.day % len(prompts)
    selected_prompt = prompts[prompt_index]

    response = model.generate_content(selected_prompt)
    
    if hasattr(response, 'parts'):
        tips = [markdown2.markdown(part.text) for part in response.parts]
    else:
        tips = [markdown2.markdown(response)] if isinstance(response, str) else [markdown2.markdown(str(response))]
    
    return tips 


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analytics', methods=['GET'])
@login_required
def analytics():
    return render_template('analytics.html')

@app.route('/upload_sheet', methods=['POST'])
@login_required
def upload_sheet():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + "_" + filename)
            file.save(file_path)
            
            description = request.form.get('description', '')

            # Save the description and file_path to the session
            session['description'] = description
            session['file_path'] = file_path

            return redirect(url_for('analyze_sheet'))
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        app.logger.error(f"Error during file upload: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during file upload.'}), 500
    
@app.route('/analyze_sheet', methods=['GET'])
@login_required
def analyze_sheet():
    try:
        file_path = session.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        description = session.get('description', '')

        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Replace unnamed columns and NaN values with empty strings
        df.columns = [col if not col.startswith('Unnamed') else '' for col in df.columns]
        df = df.fillna('')

        # Determine the figure size dynamically
        num_rows, num_columns = df.shape

        if num_columns == 0 or num_rows == 0:
            return jsonify({'error': 'DataFrame is empty or has no valid data'}), 400

        width_per_col = 2  # width per column in inches
        height_per_row = 0.5  # height per row in inches
        max_width = 20  # max width in inches
        max_height = 20  # max height in inches

        fig_width = min(max_width, num_columns * width_per_col)
        fig_height = min(max_height, num_rows * height_per_row)

        # Render the DataFrame as an image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

        # Save the image
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Save the image path to the session
        session['image_path'] = image_filename

        # Open the image for processing
        image_open = PIL.Image.open(image_path)

        # Process the data and image for the model
        try:
            prompt = [image_open, f'Analyze this spreadsheet with the following description: {description}']
            response = model.generate_content(prompt)

            if hasattr(response, 'parts'):
                final_output = ''.join(markdown2.markdown(part.text) for part in response.parts)
            else:
                final_output = markdown2.markdown(response) if isinstance(response, str) else markdown2.markdown(str(response))

        except Exception as model_error:
            app.logger.error(f"Error from model.generate_content: {model_error}", exc_info=True)
            final_output = 'An error occurred while generating the output. Please try again later.'

        # Save the final output to the session
        session['final_output'] = final_output

        return redirect(url_for('show_results'))

    except Exception as e:
        app.logger.error(f"Error analyzing sheet: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing the sheet. Please try again later.'}), 500


@app.route('/show_results', methods=['GET'])
@login_required
def show_results():
    final_output = session.get('final_output', 'No output available.')
    image_filename = session.get('image_path')
    description = session.get('description', 'No description provided.')
    if image_filename:
        image_url = url_for('uploaded_file', filename=image_filename)
    else:
        image_url = None
    return render_template('results.html', summary=final_output, image_url=image_url, description=description)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    if request.method == 'POST':
        try:
            inventory_data = request.form.get('inventory_data', '[]')
            inventory_data = json.loads(inventory_data)
            inventory_storage[current_user.id] = inventory_data
            flash("Inventory saved successfully!", "success")
        except Exception as e:
            logger.error(f"Error saving inventory: {str(e)}")
            flash("An error occurred while saving your inventory. Please try again.", "danger")
        return redirect(url_for("inventory"))

    inventory = inventory_storage.get(current_user.id, [])
    return render_template('inventory.html', inventory=inventory)

@app.route('/expense_tracker')
@login_required
def expense_tracker():
    return render_template('expense_tracker.html')

@app.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    try:
        date = request.form['date']
        amount = float(request.form['amount'])
        category = request.form['category']
        type = request.form['type']

        transaction = {
            'date': date,
            'amount': amount,
            'category': category,
            'type': type
        }

        transactions.append(transaction)
        return redirect(url_for('expense_tracker'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_transactions', methods=['GET'])
@login_required
def get_transactions():
    return jsonify(transactions)

@app.route('/calendar')
@login_required
def calendar():
    return render_template('calendar.html')
@app.route('/employees', methods=['GET', 'POST'])
@login_required
def employees_func():
    if request.method == 'POST':
        employee = {
            'id': str(uuid.uuid4()),
            'name': request.form['name'],
            'role': request.form['role'],
            'strengths': request.form['strengths'],
            'skills': request.form['skills']
        }
        employees.append(employee)
        return redirect(url_for('employees_func'))
    return render_template('employees.html', employees=employees)

@app.route('/task_creation', methods=['GET', 'POST'])
@login_required
def task_creation():
    if request.method == 'POST':
        if 'select_employee' in request.form:
            # Handle task assignment
            task = {
                'id': str(uuid.uuid4()),
                'title': request.form['title'],
                'description': request.form['description'],
                'due_date': request.form['due_date'],
                'assigned_to': request.form['assigned_to']
            }
            tasks.append(task)
            return redirect(url_for('task_creation'))
        else:
            # Generate employee suggestion
            prompt = f"Find the best employee for this task: {request.form['description']}. Employees: {employees}"
            response = model.generate_content(prompt)  # Assuming model is already defined and initialized

            # Handling response correctly
            if hasattr(response, 'parts'):
                suggested_employee = ''.join([markdown2.markdown(part.text) for part in response.parts])
            else:
                suggested_employee = markdown2.markdown(response) if isinstance(response, str) else markdown2.markdown(str(response))
            
            # Render the template with the suggested employee
            return render_template('task_creation.html', employees=employees, suggested_employee=suggested_employee, task_details=request.form)
    
    # GET request: Render the form with no prefilled data
    return render_template('task_creation.html', employees=employees, tasks=tasks)





@app.route('/dashboard_data', methods=['GET'])
@login_required
def dashboard_data():
    today = datetime.today()
    last_30_days = today - timedelta(days=30)

    # Filter last 30 days transactions
    last_30_days_transactions = [t for t in transactions if datetime.strptime(t['date'], '%Y-%m-%d') >= last_30_days]

    # Split income and expenses
    expenses = [{'date': t['date'], 'value': t['amount']} for t in last_30_days_transactions if t['type'] == 'expense']
    income = [{'date': t['date'], 'value': t['amount']} for t in last_30_days_transactions if t['type'] == 'income']


    return jsonify({
        'expenses': expenses,
        'income': income,
        'tasks': tasks
    })


if __name__ == "__main__":
    app.run(ssl_context=('cert.pem', 'key.pem'), host='0.0.0.0', port=5000)

