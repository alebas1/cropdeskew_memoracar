"""
Date: 24/02/2021
Author: Axel Lebas
File: app.py

Instantiate the Flask app.
Implement the development server.
"""

from flask import Flask, send_file
import config
from flask import request, abort
from scanDeskew.main import scan_and_deskew

app = Flask(__name__)


@app.route('/scanDeskew', methods=['PUT'])
def scan_deskew():
    """
    Attributes:
        token: authorization token (static for now)
        uploaded_file: the file sent in the request (file to scan)
        scanned_file: scanned and de-skewed file
    :return: a 200 response with a binary output (the scanned_file)
    """
    token = request.headers.get('Authorization')

    if token is None:
        abort(401, description='Request must be authenticated.')

    if token != config.validTokenExample:
        abort(403, description="Invalid token.")

    uploaded_file = request.files['image']

    if not uploaded_file:
        app.logger.error("Cannot process file, filename is empty")

    app.logger.info(f'filename: ${uploaded_file.filename}')

    scanned_file = scan_and_deskew(uploaded_file)

    return send_file(scanned_file, mimetype='image/jpg')


if __name__ == '__main__':
    app.run()

"""
curl -i -X PUT -H "Authorization: 8bf54562-3000-11eb-adc1-0242ac120002" -F "image=@/home/axel/Pictures/DS_MemoraCar/greyCard-1.jpg" http://localhost:5000/scanDeskew
"""