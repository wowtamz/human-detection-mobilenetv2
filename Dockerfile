# you can safely ignore this file for the course
# this dockerfile is used for the test runner
# you do not need docker for this course

# use just a standard python container, no CUDA will be available for the tests
FROM python:3.10
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN rm requirements.txt
