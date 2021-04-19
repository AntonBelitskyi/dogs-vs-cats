# DOGS VS CATS

## Init project

### First Step
#### Download project
`git clone http://gitlab.groupbwt.com/belitskyi-av/dogs-vs-cats.git`
### Second step
#### Install python requirements

`pip install -r requirements.txt`

## Usage

### First mode: using API

#### Run flask application

`python runserver.py`

#### Make request
 - HTTP Method: `POST`
 - URL: `http://0.0.0.0:5000/api/v1/predict/cats_and_dogs/`
 - Body: `{'image': 'your image file'}`

### Second mode: using 'main.py' script
#### Create 'logs' directory in main folder
`mkdir logs`
#### Run script
`python main.py`
 - `-i, --input-path` - path to your image or directory with images
 - `-pm, --predict-model` - name of predict model (ex. ```cats_and_dogs```)

#### Check results
All results are saved in ```logs/<predict model name>.log```

You can run ```tail -f logs/<predict model name>.log``` in another terminal 
to observe the result in real time.

## Run Tests
`pytest -v`