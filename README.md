# Fake-News-Detection
This is fake news detection app with 85% accuracy.

## Heroku deployment
While we deploy our application we use gunicorn instead of flask server.
###### Why do we use gunicorn and not the flask server?
Because the flask server we have used is not multi-threaded and so it's not scalable to be used on any production environment.

--> And we should attach our requirements.txt file with our project otherwise heroku can't find the Buildpacks. 