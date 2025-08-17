from flask import Flask, render_template, request, url_for, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from datetime import datetime
import os
import json
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "supersecretkey"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ADD THIS: JSON filter for templates
@app.template_filter('from_json')
def from_json_filter(value):
    """Parse JSON string into Python object"""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON data"}

# Import your router components (simplified for integration)
OPENROUTER_API_KEY = os.getenv('LLM_ROUTER')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

class RouterHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    classification = db.Column(db.String(250), nullable=False)
    recommended_model = db.Column(db.String(250), nullable=False)
    priority = db.Column(db.String(250), nullable=False)
    response = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('router_history', lazy=True))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Simplified router logic for web integration
class SimpleRouter:
    def __init__(self):
        self.model_mapping = {
            "simple_qa": {
                "cost": "deepseek/deepseek-r1:free",
                "speed": "deepseek/deepseek-r1:free",
                "quality": "deepseek/deepseek-r1:free"
            },
            "complex_analysis": {
                "cost": "google/gemini-pro",
                "speed": "anthropic/claude-3-haiku", 
                "quality": "openai/gpt-4"
            },
            "code_generation": {
                "cost": "mistralai/mistral-7b-instruct",
                "speed": "anthropic/claude-3-haiku",
                "quality": "mistralai/mistral-7b-instruct"
            },
            "creative_writing": {
                "cost": "anthropic/claude-3-haiku",
                "speed": "anthropic/claude-3-haiku",
                "quality": "mistralai/mistral-7b-instruct"
            },
            "summarization": {
                "cost": "mistralai/mistral-7b-instruct",
                "speed": "anthropic/claude-3-haiku",
                "quality": "openai/gpt-3.5-turbo"
            },
            "translation": {
                "cost": "openai/gpt-3.5-turbo",
                "speed": "mistralai/mistral-7b-instruct",
                "quality": "openai/gpt-4"
            },
            "math": {
                "cost": "google/gemini-pro",
                "speed": "mistralai/mistral-7b-instruct",
                "quality": "openai/gpt-4"
            }
        }
    
    def classify_task(self, prompt):
        """Classify the type of task based on the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["code", "programming", "function", "script", "python", "javascript", "coding", "debug"]):
            return "code_generation"
        elif any(word in prompt_lower for word in ["story", "creative", "fiction", "poem", "narrative", "writing", "character"]):
            return "creative_writing"
        elif any(word in prompt_lower for word in ["summary", "summarize", "brief", "condense", "tldr"]):
            return "summarization"
        elif any(word in prompt_lower for word in ["what is", "define", "explain", "simple", "quick question"]):
            return "simple_qa"
        elif any(word in prompt_lower for word in ["analysis", "analyze", "research", "complex", "detailed", "examine"]):
            return "complex_analysis"
        elif any(word in prompt_lower for word in ["translate", "translation", "language"]):
            return "translation"
        elif any(word in prompt_lower for word in ["math", "calculate", "equation", "solve", "formula"]):
            return "math"
        else:
            return "simple_qa"  # Default
    
    def recommend_model(self, task_type, priority="cost"):
        """Get model recommendation based on task type and priority"""
        return self.model_mapping.get(task_type, self.model_mapping["simple_qa"]).get(priority, "mistralai/mistral-7b-instruct")
    
    def make_request(self, prompt, model, max_tokens=1000):
        """Make API request to OpenRouter"""
        if not OPENROUTER_API_KEY:
            return {"error": "OpenRouter API key not configured"}
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "content": result['choices'][0]['message']['content'],
                    "usage": result.get('usage', {})
                }
            else:
                return {"error": f"API Error {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

router = SimpleRouter()

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("chat"))
    return render_template("home.html")

@app.route("/chat")
@login_required
def chat():
    history = RouterHistory.query.filter_by(user_id=current_user.id).order_by(RouterHistory.created_at.desc()).limit(10).all()
    return render_template("chat.html", history=history)

@app.route("/process_prompt", methods=["POST"])
@login_required
def process_prompt():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    priority = data.get("priority", "cost")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Step 1: Classify the task
    classification = router.classify_task(prompt)
    
    # Step 2: Get model recommendation
    recommended_model = router.recommend_model(classification, priority)
    
    # Step 3: Make the API request
    api_response = router.make_request(prompt, recommended_model)
    
    # Step 4: Save to database
    router_entry = RouterHistory(
        user_id=current_user.id,
        prompt=prompt,
        classification=classification,
        recommended_model=recommended_model,
        priority=priority,
        response=json.dumps(api_response)
    )
    db.session.add(router_entry)
    db.session.commit()
    
    # Step 5: Return response
    return jsonify({
        "classification": classification,
        "recommended_model": recommended_model,
        "priority": priority,
        "api_response": api_response,
        "routing_explanation": f"Classified as '{classification}' task with '{priority}' priority. Routed to {recommended_model} for optimal performance."
    })

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if User.query.filter_by(username=username).first():
            return render_template("sign_up.html", error="Username already taken!")
        
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for("login"))
        
    return render_template("sign_up.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("chat"))
        else:
            return render_template("login.html", error="Invalid username or password")
    
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)