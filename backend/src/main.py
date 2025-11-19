# import os
# import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# from flask import Flask, jsonify

# app = Flask(__name__)

# Uncomment the following line if you need to use mysql, do not modify the SQLALCHEMY_DATABASE_URI configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"


# @app.route("/health", methods=["GET"])
# def health_check():
#    """Health check endpoint"""
##     return jsonify({"status": "ok"}), 200
#
#
## if __name__ == "__main__":
#    # Listen on all interfaces, important for accessibility when exposing port
##     app.run(host="0.0.0.0", port=5000, debug=True)
