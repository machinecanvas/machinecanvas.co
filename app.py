from flask import (
    Flask, render_template, request, redirect, url_for, session, flash,
    send_file, abort, jsonify
)
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime, timedelta
import os
import uuid
import re
import requests
import MySQLdb.cursors
import smtplib
from PIL import Image, ImageDraw, ImageFont
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
import stripe
import logging
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer

app = Flask(__name__, static_folder='static')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Get SECRET_KEY from the .env file, or use a default value for development
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')

# Initialize URLSafeTimedSerializer
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# Flask Application Configuration
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:14031949@localhost/machinecanvas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# MySQL Configuration
app.config.update(
    MYSQL_HOST=os.getenv('MYSQL_HOST', 'localhost'),
    MYSQL_USER=os.getenv('MYSQL_USER', 'root'),
    MYSQL_PASSWORD=os.getenv('MYSQL_PASSWORD', '14031949'),
    MYSQL_DB=os.getenv('MYSQL_DB', 'machinecanvas'),
    MYSQL_CURSORCLASS='DictCursor'
)
mysql = MySQL(app)

# Email Configuration
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME=os.getenv('MAIL_USERNAME', 'machinecanvas.co@gmail.com'),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD', 'lnhp ivha wuxn duor'),
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_USERNAME', 'machinecanvas.co@gmail.com')
)
mail = Mail(app)

# Upload Folder Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join('static', 'uploads'),
    TEMP_UPLOAD_FOLDER=os.path.join(os.getcwd(), 'temp'),
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'webp'}
)
os.makedirs(app.config['TEMP_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables from the .env file
load_dotenv()

# Retrieve PayPal credentials from .env


PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")
PAYPAL_API_URL = os.getenv("PAYPAL_API_URL", "https://api-m.sandbox.paypal.com")


# Stripe Configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# Initialize URLSafeTimedSerializer
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Categories and Folder Creation
categories = [
    'abstract', 'nature', 'animals', 'technology', 'people', 'architecture',
    'art-design', 'business', 'fashion', 'food-drink', 'vehicles',
    'home-decor', 'animated', 'sports', 'world', 'history', 'finance', 'miscellaneous'
]
for category in categories:
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], category), exist_ok=True)

# Helper Functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_confirmation_token(email):
    """Generate a confirmation token for email."""
    return serializer.dumps(email, salt='email-confirmation-salt')

def verify_confirmation_token(token, expiration=3600):
    """Verify the email confirmation token."""
    try:
        return serializer.loads(token, salt='email-confirmation-salt', max_age=expiration)
    except Exception:
        return None

def generate_password_reset_token(email):
    """Generate a password reset token."""
    return serializer.dumps(email, salt='password-reset-salt')

def verify_password_reset_token(token, expiration=3600):
    """Verify the password reset token."""
    try:
        return serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    user_id = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    subscription = db.Column(db.String(50), default='free')
    confirmed = db.Column(db.Boolean, default=False)
    credits = db.Column(db.Integer, default=3)
    credits_remaining = db.Column(db.Integer, default=3)
    upload_limit = db.Column(db.Integer, default=3)
    images = db.relationship('Image', backref='user', lazy=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.user_id} - {self.email}>"

class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(120), nullable=False)
    price = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='active')

    def __repr__(self):
        return f"<Image {self.filename} - Category: {self.category}>"

# Create database tables
with app.app_context():
    db.create_all()

# Route for Home
@app.route('/', endpoint='home_page')
def home():
    print(f"Session data: {session}")  # Print all session data for debugging
    return render_template('index.html', email=session.get('email'))

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    user_id_error = None
    if request.method == 'POST':
        email = request.form['email']
        user_id = request.form['user_id']
        password = request.form['password']

        cursor = mysql.connection.cursor()

        # Check if email or user_id already exists
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account_by_email = cursor.fetchone()

        cursor.execute('SELECT * FROM users WHERE user_id = %s', (user_id,))
        account_by_user_id = cursor.fetchone()

        # Error handling for duplicate entries
        if account_by_email:
            error = 'This email is already registered. <a href="/login">Login</a>'
        elif account_by_user_id:
            user_id_error = 'This username (User ID) is already taken. Please choose another one.'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            error = 'Invalid email address!'
        elif not email or not password or not user_id:
            error = 'Please fill out all required fields!'

        # If no errors, hash password and insert into the database
        if not error and not user_id_error:
            try:
                hashed_password = generate_password_hash(password)
                # Insert the user and set to free plan
                cursor.execute(
                    '''
                    INSERT INTO users (email, user_id, password, subscription, confirmed, credits) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ''', 
                    (email, user_id, hashed_password, 'free', False, 3)
                )
                
                # Fetch the new user's ID to link to their free plan
                new_user_id = cursor.lastrowid
                
                # Insert into user_subscriptions with 3 credits and 3 upload limit for free plan
                cursor.execute(
                    '''
                    INSERT INTO user_subscriptions (user_id, plan_id, upload_limit, credits_remaining)
                    VALUES (%s, %s, %s, %s)
                    ''',
                    (new_user_id, 1, 3, 3)  # Plan ID 1 assumed to be the free plan
                )
                
                mysql.connection.commit()

                # Send confirmation email
                token = generate_confirmation_token(email)
                confirm_url = url_for('confirm_email', token=token, _external=True)
                msg = Message("Please confirm your email", recipients=[email])
                msg.body = f"Hi,\n\nPlease click the link to confirm your email: {confirm_url}"

                mail.send(msg)
                flash('A confirmation email has been sent.', 'success')

            except Exception as e:
                mysql.connection.rollback()
                flash('An error occurred during registration. Please try again.', 'danger')
                print(f"Error during registration: {e}")
            
            finally:
                cursor.close()

            return redirect(url_for('login'))

    return render_template('signup.html', error=error, user_id_error=user_id_error)


# Email Confirmation
@app.route('/confirm/<token>')
def confirm_email(token):
    # Attempt to verify the token and retrieve the email
    email = verify_confirmation_token(token)
    if not email:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    # Connect to the database and check if the user exists
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    account = cursor.fetchone()

    # Handle cases based on whether the account exists and its confirmation status
    if account:
        if account['confirmed']:
            flash('Account already confirmed. Please log in.', 'success')
        else:
            try:
                # Update the account's confirmed status
                cursor.execute('UPDATE users SET confirmed = %s WHERE email = %s', (True, email))
                mysql.connection.commit()
                flash('You have confirmed your account. Please log in.', 'success')
            except Exception as e:
                mysql.connection.rollback()
                flash('An error occurred during confirmation. Please try again.', 'danger')
                print(f"Error during email confirmation: {e}")
            finally:
                cursor.close()
    else:
        flash('No account found with that email. Please sign up.', 'warning')

    return redirect(url_for('login'))


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    login_error = None  # Initialize login_error variable to store specific error messages

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            login_error = 'Please provide both email and password.'
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            account = cursor.fetchone()
            cursor.close()

            if account:
                if not account['confirmed']:
                    login_error = 'Please confirm your email first.'
                elif check_password_hash(account['password'], password):
                    # Store username (user_id) and primary key ID in the session
                    session['user_id'] = account['id']  # Primary key from `users` table
                    session['username'] = account['user_id']  # Username from `users` table
                    session['email'] = account['email']
                    session['logged_in'] = True  # Additional flag to indicate logged-in status
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('home_page'))
                else:
                    login_error = 'Incorrect email/password!'
            else:
                login_error = 'No account found. Please sign up.'

    # Pass login_error to the template for display under the email field
    return render_template('login.html', login_error=login_error)

@app.context_processor
def inject_categories():
    upload_limit = None
    credits_remaining = None

    if 'user_id' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Attempt to retrieve user subscription details
        cursor.execute("""
            SELECT upload_limit, credits_remaining 
            FROM user_subscriptions 
            WHERE user_id = %s
        """, (session['user_id'],))  # session['user_id'] now stores an integer
        
        user_data = cursor.fetchone()
        cursor.close()
        
        if user_data:
            upload_limit = user_data.get('upload_limit', 3)
            credits_remaining = user_data.get('credits_remaining', 3)
            print(f"User {session['user_id']} - Injected upload_limit: {upload_limit}, credits_remaining: {credits_remaining}")
        else:
            print("User logged in but no subscription data found; defaults will apply.")

    categories = [
        "abstract", "nature", "animals", "technology", "people", "architecture",
        "art_design", "business", "fashion", "food_drink", "vehicles",
        "home_decor", "animated", "sports", "world", "history", "finance", "miscellaneous"
    ]
    
    return dict(categories=categories, upload_limit=upload_limit, credits_remaining=credits_remaining)


# Password Reset Routes
@app.route('/forgotten-password', methods=['GET', 'POST'])
def forgotten_password():
    if request.method == 'POST':
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            token = generate_password_reset_token(email)
            reset_link = url_for('reset_password', token=token, _external=True)

            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f"To reset your password, click the following link: {reset_link}"

            try:
                mail.send(msg)
                flash('A password reset link has been sent to your email address.', 'success')
            except Exception as e:
                flash('Failed to send the email. Please try again later.', 'danger')
        else:
            flash('No account with that email found.', 'danger')

    return render_template('forgotten_password.html')

@app.route('/')
def index():
    email = session.get('email') if 'loggedin' in session else None  # Check if the user is logged in
    return render_template('index.html', email=email)

# Route for Resetting Password
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        # Verify and decode the token
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except Exception:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('forgotten_password'))

    if request.method == 'POST':
        new_password = request.form['new_password']

        # Hash the new password
        hashed_password = generate_password_hash(new_password)

        # Update the password in the database
        try:
            cursor = mysql.connection.cursor()
            cursor.execute('UPDATE users SET password = %s WHERE email = %s', (hashed_password, email))
            mysql.connection.commit()
            cursor.close()

            flash('Your password has been updated. You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('An error occurred while updating your password. Please try again.', 'danger')
            logging.error(f"Error resetting password: {e}")
            return redirect(url_for('forgotten_password'))

    return render_template('reset_password.html', token=token)

# Pricing Route for Viewing, Subscribing, and Upgrading Plans
@app.route('/pricing', methods=['GET', 'POST'])
def pricing():
    cursor = mysql.connection.cursor(DictCursor)  # Use DictCursor for dictionary-style results

    if request.method == 'POST':
        # Check if the user is attempting to upgrade or subscribe
        if 'upgrade' in request.form or 'subscribe' in request.form:
            # Get the plan ID from form data
            plan_id = request.form.get('plan_id')
            user_id = session.get('user_id')
            
            if not user_id:
                flash('Please log in to manage your subscription.', 'danger')
                return redirect(url_for('login'))

            # Fetch the plan details to get upload limits and other details
            cursor.execute("SELECT upload_limit, plan_name FROM pricing_plans WHERE id = %s", (plan_id,))
            plan = cursor.fetchone()

            if not plan:
                flash('Selected plan does not exist.', 'danger')
                cursor.close()  # Close the cursor before redirecting
                return redirect(url_for('pricing'))

            # Check if the user already has a subscription
            cursor.execute("SELECT * FROM user_subscriptions WHERE user_id = %s", (user_id,))
            subscription = cursor.fetchone()

            try:
                if subscription:
                    # Update existing subscription
                    cursor.execute("""
                        UPDATE user_subscriptions 
                        SET plan_id = %s, upload_limit = %s, credits_remaining = %s, uploads_this_month = 0
                        WHERE user_id = %s
                    """, (plan_id, plan['upload_limit'], plan['upload_limit'], user_id))
                    flash(f'Your subscription has been updated to the {plan["plan_name"]} plan!', 'success')
                else:
                    # Create a new subscription
                    cursor.execute("""
                        INSERT INTO user_subscriptions (user_id, plan_id, upload_limit, credits_remaining)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id, plan_id, plan['upload_limit'], plan['upload_limit']))
                    flash(f'You have successfully subscribed to the {plan["plan_name"]} plan!', 'success')

                mysql.connection.commit()
            except Exception as e:
                mysql.connection.rollback()  # Rollback in case of an error
                flash('An error occurred while processing your subscription. Please try again.', 'danger')
                print(f"Subscription error: {e}")
            finally:
                cursor.close()

            return redirect(url_for('manage_subscription'))

    # Handle GET request to show available pricing plans
    cursor.execute("SELECT id, plan_name, price, upload_limit, revenue_share, description FROM pricing_plans")
    plans = cursor.fetchall()
    cursor.close()

    # Render the pricing page with the available plans
    return render_template('pricing.html', plans=plans)


from PIL import Image
import os

def is_ai_image(filepath):
    """
    A heuristic-based function to check if an image might be AI-generated.
    Uses basic criteria like file size and image dimensions to make a determination.
    This method is intended as a temporary solution and may not be highly accurate.
    
    Args:
        filepath (str): The path to the image file.

    Returns:
        bool: True if the image is likely AI-generated, False otherwise.
    """
    try:
        # Check file size (e.g., most generative images are larger in size)
        file_size = os.path.getsize(filepath)
        if file_size > 2 * 1024 * 1024:  # Example threshold: size > 2MB
            return True

        # Check image dimensions (e.g., common resolutions for generated images)
        with Image.open(filepath) as img:
            width, height = img.size
            # Example dimensions for generated images: divisible by 64 or larger than 1024x1024
            if (width % 64 == 0 and height % 64 == 0) or (width > 1024 and height > 1024):
                return True

    except Exception as e:
        # Log any exceptions encountered during the image check
        print(f"Error processing image for AI detection: {e}")
        return False  # Assume non-AI if there’s an issue with reading the image

    # Default to False if no AI-indicative criteria are met
    return False


from werkzeug.utils import secure_filename
import os

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    os.makedirs(app.config['TEMP_UPLOAD_FOLDER'], exist_ok=True)

    if 'user_id' not in session:
        flash('You must be logged in to upload!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    unlimited_credits = (user_id == 19)  # Enable unlimited credits for user_id 19

    if request.method == 'POST':
        file = request.files.get('file')
        category = request.form.get('category')
        price = request.form.get('price')

        if not (file and allowed_file(file.filename) and category and price):
            flash('Please complete all fields and use a valid file type.', 'danger')
            return redirect(url_for('upload_image'))

        # Secure file saving paths
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], category)
        final_path = os.path.join(upload_path, filename)
        os.makedirs(upload_path, exist_ok=True)

        try:
            # Save the file temporarily
            file.save(temp_path)

            # Check AI validation
            if not is_ai_image(temp_path):
                os.remove(temp_path)  # Clean up temp file
                flash('Only AI-generated images are allowed.', 'danger')
                return redirect(url_for('upload_image'))

            # Check subscription details if not unlimited credits
            cursor = mysql.connection.cursor()
            if not unlimited_credits:
                cursor.execute("""
                    SELECT upload_limit, uploads_this_month, credits_remaining 
                    FROM user_subscriptions WHERE user_id = %s
                """, (user_id,))
                user_subscription = cursor.fetchone()

                if user_subscription:
                    upload_limit = user_subscription.get('upload_limit', 0)
                    uploads_this_month = user_subscription.get('uploads_this_month', 0)
                    credits_remaining = user_subscription.get('credits_remaining', 0)

                    if uploads_this_month >= upload_limit:
                        os.remove(temp_path)  # Clean up temp file
                        flash('You have reached your upload limit for this month.', 'danger')
                        return redirect(url_for('upload_image'))

                    if credits_remaining <= 0:
                        os.remove(temp_path)  # Clean up temp file
                        flash('Need credits to upload. Please visit the pricing page.', 'danger')
                        return redirect(url_for('upload_image'))

            # Ensure unique filename
            counter = 1
            while os.path.exists(final_path):
                final_path = os.path.join(upload_path, f"{os.path.splitext(filename)[0]}_{counter}{os.path.splitext(filename)[1]}")
                counter += 1

            # Move file to final location
            os.rename(temp_path, final_path)

            # Insert image details into the database
            cursor.execute(
                'INSERT INTO images (filename, category, price, user_id) VALUES (%s, %s, %s, %s)',
                (os.path.basename(final_path), category, price, user_id)
            )
            mysql.connection.commit()
            image_id = cursor.lastrowid  # Retrieve last inserted image ID

            # Apply watermark
            try:
                add_watermark(final_path, watermark_text="Your Watermark Text")
            except Exception as e:
                print(f"Watermarking failed: {e}")
                flash('An error occurred while adding the watermark.', 'warning')

            # Update subscription details if applicable
            if not unlimited_credits:
                cursor.execute("""
                    UPDATE user_subscriptions 
                    SET uploads_this_month = uploads_this_month + 1, 
                        credits_remaining = credits_remaining - 1
                    WHERE user_id = %s
                """, (user_id,))
                mysql.connection.commit()

            flash('AI-generated image uploaded successfully!', 'success')
            return redirect(url_for('category_page', category_name=category))

        except Exception as e:
            print(f"Upload Error: {e}")
            flash('An error occurred. Please try again.', 'danger')
        finally:
            if os.path.exists(temp_path):  # Clean up temp file
                os.remove(temp_path)
            if 'cursor' in locals():
                cursor.close()

    return render_template('upload.html', user_id=user_id)



from PIL import Image, ImageDraw, ImageFont

# Add watermark to image
def add_watermark(image_path, watermark_text="Sample Watermark"):
    """
    Adds a watermark to an image.
    """
    with Image.open(image_path).convert("RGBA") as base:
        width, height = base.size
        watermark_overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))

        try:
            # Use a basic font with a specific size
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            # Fall back to the default font if the specified font is unavailable
            font = ImageFont.load_default()

        # Get the bounding box of the text to calculate text width and height
        text_bbox = font.getbbox(watermark_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (width - text_width - 10, height - text_height - 10)

        draw = ImageDraw.Draw(watermark_overlay)
        draw.text(text_position, watermark_text, font=font, fill=(255, 255, 255, 128))

        watermarked_image = Image.alpha_composite(base, watermark_overlay)
        watermarked_image = watermarked_image.convert("RGB")
        watermarked_image.save(image_path, "JPEG")



@app.route('/save_image/<int:image_id>', methods=['POST'])
def save_image(image_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify(success=False, message="Please log in to save images."), 401

    cursor = mysql.connection.cursor()
    
    # Check if image_id exists in the images table
    cursor.execute("SELECT id FROM images WHERE id = %s", (image_id,))
    image_exists = cursor.fetchone()
    
    if not image_exists:
        return jsonify(success=False, message="The selected image does not exist."), 404

    # Insert the image into the 'saved_images' table
    cursor.execute("""
        INSERT INTO saved_images (user_id, image_id)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE image_id = image_id
    """, (user_id, image_id))
    mysql.connection.commit()
    cursor.close()

    return jsonify(success=True, message="Image saved successfully!")


@app.route('/saved_images')
def saved_images():
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view your saved images.", "warning")
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT images.id, images.filename, images.category, images.price 
        FROM images 
        INNER JOIN saved_images ON images.id = saved_images.image_id 
        WHERE saved_images.user_id = %s
    """, (user_id,))
    saved_images = cursor.fetchall()  # Fetch all rows where images are saved by this user
    cursor.close()

    return render_template('saved_images.html', saved_images=saved_images)

@app.route('/delete_saved_image/<int:image_id>', methods=['POST'])
def delete_saved_image(image_id):
    user_id = session.get('user_id')
    if not user_id:
        flash("You need to be logged in to remove saved images.", "danger")
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    try:
        # Delete the saved image entry for the current user and image_id
        cursor.execute("DELETE FROM saved_images WHERE user_id = %s AND image_id = %s", (user_id, image_id))
        mysql.connection.commit()
        flash("Image removed from saved images.", "success")
    except Exception as e:
        flash("An error occurred while removing the image.", "danger")
        print(f"Error removing image: {e}")
    finally:
        cursor.close()

    return redirect(url_for('saved_images'))


from MySQLdb.cursors import DictCursor

@app.route('/category/<category_name>', endpoint='category_page')
def category_page(category_name):
    cursor = mysql.connection.cursor(DictCursor)  # Use DictCursor for dictionary-style access
    cursor.execute('SELECT id, filename, category, price FROM images WHERE category = %s', (category_name,))
    category_images = cursor.fetchall()
    cursor.close()

    # Check if there are images in the category and pass additional context
    image_count = len(category_images)
    if image_count == 0:
        flash(f"No images available in the {category_name.replace('_', ' ').title()} category.", 'info')

    # Render the generic 'category.html' template with context
    return render_template(
        'category.html', 
        category_name=category_name, 
        category_images=category_images,
        image_count=image_count
    )


@app.route('/profile/settings', methods=['GET', 'POST'], endpoint='profile_settings')
def profile_settings():
    # Check if the user is logged in
    if 'email' not in session:
        flash('Please log in to access settings.', 'danger')
        return redirect(url_for('login'))

    email = session['email']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Fetch user info based on email
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    
    # Ensure user exists
    if not user:
        cursor.close()
        flash('User not found.', 'danger')
        return redirect(url_for('login'))

    # Fetch credits_remaining from user_subscriptions
    cursor.execute("""
        SELECT us.credits_remaining 
        FROM user_subscriptions us
        JOIN users u ON u.id = us.user_id
        WHERE u.email = %s
    """, (email,))
    subscription = cursor.fetchone()
    
    # Extract credits remaining, or set to None if no subscription is found
    credits_remaining = subscription['credits_remaining'] if subscription else None

    if request.method == 'POST':
        # Handle password change
        current_password = request.form['current_password']
        new_password = request.form['new_password']

        # Validate current password
        if check_password_hash(user['password'], current_password):
            # Update password
            hashed_new_password = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_new_password, email))
            mysql.connection.commit()
            flash('Password updated successfully!', 'success')
        else:
            flash('Current password is incorrect.', 'danger')
        
        cursor.close()
        return redirect(url_for('profile_settings'))

    cursor.close()
    # Render the profile settings template with user and credits information
    return render_template('settings.html', user=user, credits_remaining=credits_remaining)


@app.route('/manage_subscription', methods=['GET', 'POST'])
def manage_subscription():
    if 'user_id' not in session:
        flash('Please log in to manage your subscription.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    cursor = mysql.connection.cursor(dictionary=True)

    # Fetch the user's current subscription details
    cursor.execute("""
        SELECT us.plan_id, us.start_date, us.credits_remaining, p.plan_name, p.price, p.upload_limit, p.revenue_share 
        FROM user_subscriptions us
        JOIN pricing_plans p ON us.plan_id = p.id
        WHERE us.user_id = %s
    """, (user_id,))
    subscription = cursor.fetchone()

    # If no subscription, redirect to the pricing page
    if not subscription:
        flash('You do not have an active subscription. Please select a plan to subscribe.', 'info')
        cursor.close()
        return redirect(url_for('manage_subscription'))

    # Handle subscription update/cancellation requests
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'cancel':
            cursor.execute("DELETE FROM user_subscriptions WHERE user_id = %s", (user_id,))
            mysql.connection.commit()
            flash('Your subscription has been canceled.', 'success')
            cursor.close()
            return redirect(url_for('manage_subscription'))
        
        elif action == 'change_plan':
            new_plan_id = request.form.get('new_plan_id')
            if new_plan_id:
                cursor.execute("UPDATE user_subscriptions SET plan_id = %s WHERE user_id = %s", (new_plan_id, user_id))
                mysql.connection.commit()
                flash('Your subscription plan has been updated.', 'success')
                return redirect(url_for('manage_subscription'))
            else:
                flash('Please select a valid plan to update.', 'danger')

    # Fetch all available plans to display as options for plan change
    cursor.execute("SELECT * FROM pricing_plans")
    available_plans = cursor.fetchall()
    cursor.close()

    return render_template(
        'manage_subscription.html',
        subscription=subscription,
        available_plans=available_plans
    )

@app.route('/profile/my-images', endpoint='profile_images')
def profile_images():
    if 'user_id' not in session:
        flash('Please log in to view your uploaded images.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT id, filename, category, price FROM images WHERE user_id = %s', (user_id,))
    uploaded_images = cursor.fetchall()
    cursor.close()
    return render_template('my_images.html', uploaded_images=uploaded_images)


@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    if 'user_id' not in session:
        flash('Please log in to delete images.', 'error')
        return redirect(url_for('login'))
    
    # Delete the image from the database and file system
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT filename, category FROM images WHERE id = %s AND user_id = %s', (image_id, session['user_id']))
    image = cursor.fetchone()

    if image:
        # Remove the image file from the filesystem
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image['category'], image['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete the record from the database
        cursor.execute('DELETE FROM images WHERE id = %s', (image_id,))
        mysql.connection.commit()
        flash('Image deleted successfully.', 'success')
    else:
        flash('Image not found or you do not have permission to delete this image.', 'error')
    
    cursor.close()
    return redirect(url_for('profile_images'))


# Route for Purchase History
@app.route('/purchase_history')
def purchase_history():
    if 'email' not in session:
        flash('You must be logged in to view your purchase history.', 'danger')
        return redirect(url_for('login'))

    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM purchases WHERE user_email = %s", (session['email'],))
        purchases = cursor.fetchall()
    except Exception as e:
        flash("An error occurred while fetching your purchase history.", 'danger')
        print(f"Error: {e}")
    finally:
        cursor.close()

    return render_template('purchase_history.html', purchases=purchases)


@app.route('/settings', endpoint='general_settings')
def general_settings():
    if 'user_id' not in session:
        flash('Please log in to access settings.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']

    cursor = mysql.connection.cursor()

    # Fetch user email and bank details
    cursor.execute("""
        SELECT 
            email, 
            account_holder_name, 
            account_number, 
            sort_code 
        FROM users 
        WHERE id = %s
    """, (user_id,))
    user_data = cursor.fetchone()

    # Build the user dictionary
    user = {
        "email": user_data[0],
        "bank_details": {
            "account_holder_name": user_data[1],
            "account_number": user_data[2],
            "sort_code": user_data[3],
        }
    }

    # Fetch subscription details
    cursor.execute("""
        SELECT p.plan_name, p.price, p.upload_limit, p.revenue_share
        FROM user_subscriptions us
        JOIN pricing_plans p ON us.plan_id = p.id
        WHERE us.user_id = %s
    """, (user_id,))
    subscription = cursor.fetchone()

    # Extract upload limit and credits remaining (if applicable)
    upload_limit = subscription.get('upload_limit') if subscription else None
    credits_remaining = subscription.get('credits_remaining') if subscription else None

    cursor.close()

    # Render the settings template with the fetched data
    return render_template(
        'settings.html',
        user=user,
        subscription=subscription,
        upload_limit=upload_limit,
        credits_remaining=credits_remaining
    )


# Banking details
@app.route('/update_bank_details', methods=['POST'])
def update_bank_details():
    if 'user_id' not in session:
        flash('Please log in to update bank details.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Fetch form data
    account_holder_name = request.form.get('account_holder_name')
    account_number = request.form.get('account_number')
    sort_code = request.form.get('sort_code')

    # Validate input
    if not account_holder_name.strip():
        flash('Account holder name is required.', 'danger')
        return redirect(url_for('profile_settings'))
    if not account_number.isdigit() or len(account_number) != 8:
        flash('Account number must be 8 digits.', 'danger')
        return redirect(url_for('profile_settings'))
    if not sort_code.isdigit() or len(sort_code) != 6:
        flash('Sort code must be 6 digits.', 'danger')
        return redirect(url_for('profile_settings'))

    try:
        # Update the bank details in the database
        cursor = mysql.connection.cursor()
        cursor.execute("""
            UPDATE users
            SET account_holder_name = %s, account_number = %s, sort_code = %s
            WHERE id = %s
        """, (account_holder_name, account_number, sort_code, user_id))
        mysql.connection.commit()
        flash('Bank details updated successfully.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash('An error occurred while updating bank details. Please try again.', 'danger')
        print(f"Error updating bank details: {e}")
    finally:
        cursor.close()

    return redirect(url_for('profile_settings'))



# delete account
@app.route('/profile/delete-account', methods=['GET', 'POST'])
def delete_account():
    if 'user_id' not in session:
        flash('Please log in to delete your account.', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['user_id']

    if request.method == 'POST':
        cursor = mysql.connection.cursor()

        try:
            cursor.execute("DELETE FROM user_subscriptions WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM images WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            mysql.connection.commit()
            flash('Account successfully deleted.', 'success')
            session.clear()  # Clear session after account deletion
        except Exception as e:
            mysql.connection.rollback()
            flash('An error occurred while deleting your account. Please try again.', 'danger')
            print(f"Error: {e}")
        finally:
            cursor.close()
        
        return redirect(url_for('signup'))
    
    # Render confirmation page on GET request
    return render_template('delete_account_confirm.html')

# Payment Route to handle individual image purchases
@app.route('/payment')
def payment():
    image = request.args.get('image')
    if image:
        # Add logic to handle image purchase, like showing a payment form or redirecting to checkout
        return render_template('payment.html', image=image)
    else:
        flash("No image selected for purchase.", "warning")
        return redirect(url_for('category_page', category_name=request.args.get('category_name', 'abstract')))

    # Define subscription plans as a dictionary
SUBSCRIPTION_PLANS = {
    "basic": {
        "name": "Basic",
        "price": 10,
        "features": [
            "Up to 10 image uploads per month",
            "Access to basic features",
            "Standard revenue share on sales"
        ]
    },
    "pro": {
        "name": "Pro",
        "price": 25,
        "features": [
            "Up to 50 image uploads per month",
            "Access to all platform features",
            "Reduced revenue share on sales"
        ]
    },
    "premium": {
        "name": "Premium",
        "price": 50,
        "features": [
            "Unlimited image uploads",
            "Priority support",
            "Reduced revenue share on sales"
        ]
    }
}

@app.route('/cart_count', methods=['GET'])
def cart_count():
    try:
        cart = session.get('cart', [])
        return jsonify({'cart_count': len(cart)})
    except Exception as e:
        print(f"Error in cart_count: {e}")
        return jsonify({'cart_count': 0}), 500



@app.route('/cart')
def cart():
    """Display the cart page with items and total amount."""
    cart_items = session.get('cart', [])
    
    # Calculate subtotals for each item and total amount
    for item in cart_items:
        try:
            item['subtotal'] = round(float(item['price']) * item.get('quantity', 1), 2)
        except (ValueError, TypeError):
            item['subtotal'] = 0.0
    
    total_amount = round(sum(item['subtotal'] for item in cart_items), 2)

    return render_template('cart.html', cart_items=cart_items, total_amount=total_amount)


@app.route('/add_subscription_to_cart/<plan>', methods=['POST'])
def add_subscription_to_cart(plan):
    """Add a subscription plan to the cart or update if it exists."""
    # Define subscription plans with thumbnails
    SUBSCRIPTION_PLANS = {
        "basic": {
            "name": "Basic",
            "price": 10.0,
            "features": ["Up to 10 image uploads per month", "Access to basic features"],
            "thumbnail": url_for('static', filename='uploads/index_images/basic.webp')  # Corrected path
        },
        "pro": {
            "name": "Pro",
            "price": 25.0,
            "features": ["Up to 50 image uploads per month", "Access to all platform features"],
            "thumbnail": url_for('static', filename='uploads/index_images/pro.webp')  # Corrected path
        },
        "premium": {
            "name": "Premium",
            "price": 50.0,
            "features": ["Unlimited image uploads", "Priority support", "Reduced revenue share on sales"],
            "thumbnail": url_for('static', filename='uploads/index_images/premium.webp')  # Corrected path
        }
    }

    # Fetch the selected plan details
    plan_details = SUBSCRIPTION_PLANS.get(plan.lower())
    if not plan_details:
        flash("Invalid subscription plan selected.", "danger")
        return redirect(url_for('cart'))

    # Prepare the subscription item
    subscription_item = {
        'id': f"sub_{plan}",
        'name': f"{plan_details['name']} Plan",
        'price': plan_details['price'],
        'quantity': 1,
        'type': 'subscription',
        'features': plan_details['features'],
        'thumbnail': plan_details['thumbnail']  # Add the thumbnail field
    }

    # Retrieve or initialize the cart
    cart = session.get('cart', [])

    # Check if the plan is already in the cart
    if not any(cart_item['id'] == subscription_item['id'] for cart_item in cart):
        cart.append(subscription_item)
        session.modified = True
        flash(f"{plan_details['name']} Plan added to cart!", 'success')
    else:
        flash(f"{plan_details['name']} Plan is already in your cart.", 'info')

    # Save the updated cart to the session
    session['cart'] = cart
    return redirect(url_for('cart'))


@app.route('/purchase_credits', methods=['GET', 'POST'])
def purchase_credits():
    """
    Display available credit packages and handle adding selected ones to the cart.
    """
    if 'user_id' not in session:
        flash('Please log in to purchase credits.', 'danger')
        return redirect(url_for('login'))

    # Define available credit packages
    credit_packages = [
        {"credits": 5, "price": 5},
        {"credits": 10, "price": 10},
        {"credits": 25, "price": 20},
        {"credits": 50, "price": 30}
    ]

    if request.method == 'POST':
        # Handle adding a credit package to the cart
        try:
            credits = int(request.form.get('credits'))
            price = float(request.form.get('price'))
        except (ValueError, TypeError):
            flash("Invalid credit package selection. Please try again.", "danger")
            return redirect(url_for('purchase_credits'))

        # Validate the selected credits package
        if not any(package['credits'] == credits and package['price'] == price for package in credit_packages):
            flash("Selected credit package does not exist.", "danger")
            return redirect(url_for('purchase_credits'))

        # Create a credit package item
        credit_item = {
            'id': f"credits_{credits}",
            'name': f"{credits} Credits Package",
            'price': price,
            'quantity': 1,
            'type': 'credits',
            'credits': credits,
            'thumbnail': url_for('static', filename=f"uploads/index_images/{credits}_credits.webp")
        }

        # Retrieve or initialize the cart
        cart = session.get('cart', [])

        # Check if the credit package is already in the cart
        for cart_item in cart:
            if cart_item['id'] == credit_item['id']:
                flash(f"{credits} Credits package is already in your cart.", 'info')
                break
        else:
            cart.append(credit_item)
            session.modified = True
            flash(f"{credits} Credits added to cart!", 'success')

        # Save the updated cart back to the session
        session['cart'] = cart
        return redirect(url_for('cart'))

    # Render the purchase credits page
    return render_template('purchase_credits.html', credit_packages=credit_packages)

from flask import jsonify  # Ensure you have this imported

import os

def find_file_with_extension(category, base_filename):
    """
    Search for a file with the specified base filename in the given category directory.
    Returns the first file found with an allowed extension or None if no file is found.
    """
    # Path to the upload folder for the given category
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], category)
    allowed_extensions = app.config['ALLOWED_EXTENSIONS']

    # Loop through allowed extensions and check if a file exists
    for extension in allowed_extensions:
        potential_file = f"{base_filename}.{extension}"
        potential_path = os.path.join(upload_path, potential_file)
        if os.path.exists(potential_path):
            return potential_file  # Return the matched file with extension

    return None  # Return None if no matching file is found

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    """
    Add an item to the cart or update its quantity if it already exists.
    Supports purchased images, subscriptions, and credits.
    """
    try:
        # Extract form data
        item_id = request.form.get('item_id')
        item_type = request.form.get('type', 'regular')  # Defaults to 'regular'
        quantity = int(request.form.get('quantity', 1))
        file_type = request.form.get('file_type', 'jpeg') if item_type == 'regular' else None
        image_size = request.form.get('image_size', 'medium') if item_type == 'regular' else None

        if not item_id:
            return jsonify({'success': False, 'message': "Invalid item details. Please try again."}), 400

        item = None

        # Handle 'regular' image items
        if item_type == 'regular':
            cursor = mysql.connection.cursor()
            cursor.execute("""
                SELECT id, filename, category, price 
                FROM images 
                WHERE id = %s
            """, (item_id,))
            image_data = cursor.fetchone()
            cursor.close()

            if not image_data:
                return jsonify({'success': False, 'message': "Image not found."}), 404

            thumbnail = url_for('static', filename=f"uploads/{image_data['category']}/{image_data['filename']}")

            item = {
                'id': image_data['id'],
                'name': f"Image {image_data['id']}",
                'price': float(image_data['price']),
                'quantity': quantity,
                'type': item_type,
                'thumbnail': thumbnail,
                'file_type': file_type,
                'image_size': image_size,
                'category': image_data['category']
            }

        # Handle 'subscription' items
        elif item_type == 'subscription':
            plan = request.form.get('plan')
            price = float(request.form.get('price', 0))

            if not plan or price <= 0:
                return jsonify({'success': False, 'message': "Invalid subscription details."}), 400

            # Construct the thumbnail path for subscriptions
            thumbnail = url_for('static', filename=f"uploads/index_images/{plan.lower()}.webp")

            item = {
                'id': f"subscription_{plan}",
                'name': f"{plan.capitalize()} Subscription",
                'price': price,
                'quantity': 1,
                'type': item_type,
                'thumbnail': thumbnail,
                'file_type': None,  # Subscriptions don't have a file type
                'image_size': None  # Subscriptions don't have an image size
            }

        # Handle 'credits' items
        elif item_type == 'credits':
            credits = int(request.form.get('credits', 0))
            price = float(request.form.get('price', 0))

            if credits <= 0 or price <= 0:
                return jsonify({'success': False, 'message': "Invalid credit package details."}), 400

            # Construct the thumbnail path for credits
            thumbnail = url_for('static', filename=f"uploads/index_images/{credits}_credits.webp")

            item = {
                'id': f"credits_{credits}",
                'name': f"{credits} Credits Package",
                'price': price,
                'quantity': 1,
                'type': item_type,
                'thumbnail': thumbnail,
                'file_type': None,
                'image_size': None
            }

        # Ensure the item was created successfully
        if not item:
            return jsonify({'success': False, 'message': "Item type not supported."}), 400

        # Retrieve or initialize the cart
        cart = session.get('cart', [])

        # Update quantity if item already exists in the cart
        for cart_item in cart:
            if cart_item['id'] == item['id']:
                cart_item['quantity'] += quantity
                session['cart'] = cart
                session.modified = True
                return jsonify({
                    'success': True,
                    'message': f"Updated quantity for {item['name']} in the cart.",
                    'cart_count': len(cart)
                })

        # Add new item to the cart
        cart.append(item)
        session['cart'] = cart
        session.modified = True

        return jsonify({
            'success': True,
            'message': f"{item['name']} added to the cart!",
            'cart_count': len(cart)
        })

    except Exception as e:
        print(f"Error in add_to_cart: {e}")
        return jsonify({'success': False, 'message': "An error occurred while adding the item to the cart."}), 500

# Route to remove an item from the cart
@app.route('/remove_from_cart/<item_id>', methods=['POST'])
def remove_from_cart(item_id):
    """
    Remove an item from the cart based on its ID and update the total price.
    """
    try:
        # Retrieve the current cart from the session
        cart = session.get('cart', [])
        
        # Filter out the item to be removed
        updated_cart = [item for item in cart if str(item['id']) != str(item_id)]
        session['cart'] = updated_cart
        session.modified = True  # Mark session as modified

        # Recalculate the new total price
        new_total = sum(item['price'] * item['quantity'] for item in updated_cart)

        return jsonify({
            'success': True,
            'message': 'Item removed from cart.',
            'cart_count': len(updated_cart),  # Updated cart count
            'new_total': round(new_total, 2)  # Return the new total rounded to 2 decimal places
        })
    except Exception as e:
        print(f"Error in remove_from_cart: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while removing the item.'}), 500


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)


# Route for Checkout
@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    """
    Checkout route for displaying cart summary and creating payment sessions for Stripe and PayPal.
    """
    cart_items = session.get('cart', [])
    if not cart_items:
        flash("Your cart is empty. Please add items before checking out.", "info")
        return redirect(url_for('cart'))

    total_amount = round(
        sum(item['price'] * item.get('quantity', 1) for item in cart_items), 2
    )
    if total_amount <= 0:
        flash("The total amount is invalid.", "danger")
        return redirect(url_for('cart'))

    # Handle GET request
    if request.method == 'GET':
        # Debugging PayPal Client ID
        logging.info(f"PayPal Client ID: {PAYPAL_CLIENT_ID}")
        return render_template(
            'checkout.html',
            cart_items=cart_items,
            total_amount=total_amount,
            paypal_client_id=PAYPAL_CLIENT_ID
        )
    
    logging.info(f"PayPal Client ID passed to template: {PAYPAL_CLIENT_ID}")


    # Handle POST request
    payment_method = request.form.get('payment_method')
    try:
        if payment_method == 'stripe':
            # Stripe payment processing
            stripe_line_items = [
                {
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': item['name'],
                            'description': (
                                f"{item.get('type', '').capitalize()} Purchase | "
                                f"File Type: {item.get('file_type', 'N/A')} | "
                                f"Size: {item.get('image_size', 'N/A')}"),
                        },
                        'unit_amount': int(item['price'] * 100),
                    },
                    'quantity': item.get('quantity', 1),
                }
                for item in cart_items
            ]

            stripe_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=stripe_line_items,
                mode='payment',
                success_url=url_for('success', _external=True),
                cancel_url=url_for('checkout', _external=True),
            )
            return redirect(stripe_session.url, code=303)

        elif payment_method == 'paypal':
            # Redirect to create PayPal order endpoint
            return redirect(url_for('create_order'))

        else:
            flash("Invalid payment method selected. Please try again.", "danger")
            return redirect(url_for('checkout'))

    except Exception as e:
        # Log the error and provide user feedback
        logging.error(f"Error during payment processing: {e}", exc_info=True)
        flash("An error occurred while processing your payment. Please try again.", "danger")
        return redirect(url_for('checkout'))


# PayPal Token Retrieval
def get_paypal_token():
    """Retrieve PayPal access token."""
    try:
        response = requests.post(
            f"{PAYPAL_API_URL}/v1/oauth2/token",
            headers={"Accept": "application/json", "Accept-Language": "en_US"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve PayPal token: {e}", exc_info=True)
        return None


# API for PayPal Order Creation
@app.route('/api/paypal/create-order', methods=['POST'])
def create_order():
    """Create a PayPal order."""
    token = get_paypal_token()
    if not token:
        return jsonify({"error": "Failed to retrieve PayPal access token."}), 500

    data = request.json
    if not data or 'totalAmount' not in data:
        return jsonify({"error": "Missing 'totalAmount' in request payload."}), 400

    try:
        response = requests.post(
            f"{PAYPAL_API_URL}/v2/checkout/orders",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json={
                "intent": "CAPTURE",
                "purchase_units": [
                    {
                        "amount": {"currency_code": "USD", "value": str(data['totalAmount'])}
                    }
                ]
            }
        )
        response.raise_for_status()
        return jsonify(response.json()), 201
    except requests.exceptions.RequestException as e:
        logging.error(f"Error creating PayPal order: {e}", exc_info=True)
        return jsonify({"error": "Failed to create PayPal order."}), 500


# API for PayPal Order Capture
@app.route('/api/paypal/capture-order', methods=['POST'])
def capture_order():
    """Capture a PayPal order after buyer approval."""
    token = get_paypal_token()
    if not token:
        return jsonify({"error": "Failed to retrieve PayPal access token."}), 500

    data = request.json
    if not data or 'orderID' not in data:
        return jsonify({"error": "Missing 'orderID' in request payload."}), 400

    try:
        response = requests.post(
            f"{PAYPAL_API_URL}/v2/checkout/orders/{data['orderID']}/capture",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
        )
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        logging.error(f"Error capturing PayPal order: {e}", exc_info=True)
        return jsonify({"error": "Failed to capture PayPal order."}), 500


@app.route('/success')
def success():
    """
    Success route after payment.
    """
    # Clear the cart after successful payment
    session['cart'] = []
    session.modified = True
    return render_template('success.html', message="Payment successful! Thank you for your purchase.")

@app.route('/cancel')
def cancel():
    """
    Cancel route if payment is not completed.
    """
    flash("Payment canceled. Your items are still in the cart.", "info")
    return redirect(url_for('cart'))

# Store tokens for short-term download links
download_links = {}


# Store tokens for short-term download links
download_links = {}

# Route to confirm purchase and redirect to download options selection page
@app.route('/confirm_purchase', methods=['POST'])
def confirm_purchase():
    image_filenames = request.form.getlist('images')  # Assuming images is a list of filenames
    user_email = session.get('email')

    if image_filenames and user_email:
        # Generate a download token and entry in download_links for each image
        links_info = []
        for filename in image_filenames:
            download_token = str(uuid.uuid4())
            download_links[download_token] = {
                'file_path': os.path.join('static/uploads', filename),
                'expires': datetime.now() + timedelta(minutes=30)
            }
            links_info.append({'filename': filename, 'download_token': download_token})

        # Store download links info temporarily in session for the selection page
        session['links_info'] = links_info
        return redirect(url_for('download.html'))
    else:
        flash("Error: Unable to confirm purchase.", "danger")
        return redirect(url_for('home_page'))


# Route to display download options for each purchased image
@app.route('/select_download_options')
def select_download_options():
    links_info = session.get('links_info', [])
    return render_template('download.html', links_info=links_info, download_options=DOWNLOAD_OPTIONS)


# Route to process selected download options
@app.route('/process_download_options', methods=['POST'])
def process_download_options():
    selected_options = {}
    for link in session.get('links_info', []):
        token = link['download_token']
        selected_format = request.form.get(f'format_{token}')
        selected_size = request.form.get(f'size_{token}')
        selected_options[token] = {
            'format': selected_format,
            'size': selected_size
        }
    session['selected_options'] = selected_options
    flash("Download options confirmed. Proceed to download.", "success")
    return redirect(url_for('download_confirmation'))


# Route to handle the actual image download based on user-selected options
@app.route('/download_image/<download_token>', methods=['POST'])
def download_image(download_token):
    # Retrieve user-selected download options if applicable
    selected_options = session.get('selected_options', {}).get(download_token)
    link_info = download_links.get(download_token)
    
    if not link_info or not selected_options:
        flash("Invalid download option or token.", "danger")
        return redirect(url_for('home_page'))

    if datetime.now() > link_info['expires']:
        del download_links[download_token]
        flash("Download link has expired.", "warning")
        return redirect(url_for('home_page'))

    # Process file path and selected format/size
    file_path = link_info['file_path']
    with Image.open(file_path) as img:
        # Resize if a specific size is selected
        if selected_options['size'] != 'original':
            width, height = DOWNLOAD_OPTIONS[selected_options['size']]['width'], DOWNLOAD_OPTIONS[selected_options['size']]['height']
            img = img.resize((width, height), Image.ANTIALIAS)

        # Select format based on user choice
        img_format = DOWNLOAD_OPTIONS[selected_options['format']].get('format', 'jpeg')
        img_io = BytesIO()
        img.save(img_io, format=img_format.upper())
        img_io.seek(0)

        return send_file(img_io, as_attachment=True, download_name=f"{os.path.basename(file_path).split('.')[0]}_{selected_options['size']}.{img_format}", mimetype=f"image/{img_format}")


# Function to send a single purchase confirmation email with multiple download links
def send_receipt_email(user_email, links_info):
    # Prepare the email body with multiple download links
    email_body = "Thank you for your purchase!\n\n"
    email_body += "Here are your purchased images:\n"

    for link in links_info:
        email_body += f"- {link['filename']}:\n  Download it here: {link['download_url']} (valid for 30 minutes)\n\n"

    email_body += "Regards,\nStock AI Images Team"

    # Set up email message
    msg = MIMEMultipart()
    msg['From'] = 'machinecanvas.co@gmail.com'
    msg['To'] = user_email
    msg['Subject'] = 'Your Purchase Receipt - Stock AI Images'
    msg.attach(MIMEText(email_body, 'plain'))

    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('machinecanvas.co@gmail.com', 'your_email_password')
            server.sendmail('machinecanvas.co@gmail.com', user_email, msg.as_string())
            print("Receipt email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


        # Define download options
DOWNLOAD_OPTIONS = {
    "small": {"width": 640, "height": 480, "format": "jpeg"},
    "medium": {"width": 1280, "height": 960, "format": "jpeg"},
    "large": {"width": 1920, "height": 1440, "format": "jpeg"},
    "png": {"format": "png"},
    "pdf": {"format": "pdf"}
}



@app.route('/logout')
def logout():
    """Clear session data and log the user out."""
    session.pop('loggedin', None)
    session.pop('email', None)
    session.pop('user_id', None)  # Ensures user_id is also cleared from the session
    session.pop('username', None) 
    flash('You have successfully logged out!', 'success')
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
    
