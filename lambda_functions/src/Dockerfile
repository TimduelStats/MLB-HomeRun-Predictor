FROM amazon/aws-lambda-python:3.12

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy function and utility code
COPY functions/ /var/task/functions/
COPY utils/ /var/task/utils/

CMD [ "functions.predict_homerun.main" ]