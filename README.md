# Data Mining Technique's on COVID-19: Corona Spreading and Recovery
With corona virus spreading quickly, we need a scalable solution that tracks realtime COVID-19 outbreak serving the users and the government. We use user generated real time data to track and predict further developments of the coronavirus. This concerns a number of Amsterdam startups that have joint forces and collaborate with the City of Amsterdam and VU University Amsterdam.

## Dependencies

When pulling the code for the first time, you'll likely need to install a bunch of dependencies. To make sure you have all the right dependencies activate the virtual environment.
```
$ source env/bin/activate
```
Or download the dependencies from the requirements.txt file.

```
$ pip3 install -r requirements.txt
```

In case you need to add dependencies, please do so by adding the dependency to the virtual environment and update the requirements.txt file.
```
$ source env/bin/activate
$ pip3 install <dependency>
$ pip3 freeze > requirements.txt
```

## Running the code
```
$ python3 src/main.py
```

## Literature & Report
All the supporting literature can be found in `docs/literature`.
The report can be found in `docs` and is written according to the LNCS guidelines.
