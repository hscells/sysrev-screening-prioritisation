# Systematic Review Screening Prioritisation

_Using learning to rank._

## Setup

 - elasticsearch version 5.3.0
 - latest java version 1.8
 - trec_eval

The experiments use the learning to rank elasticsearch plugin from:
 
https://github.com/o19s/elasticsearch-learning-to-rank

To build the plugin, ensure you have the `JAVA_HOME` environment variable set, and have at least java version 1.8u40. 
On macOS, the to set `JAVA_HOME`, use `export JAVA_HOME=$(/usr/libexec/java_home)`.

Inside the `ltr-query` submodule, run:
 
```bash
./gradlew run#installLtrQueryPlugin
```

This will generate a zip file that can be installed to elasticsearch as a module. To install into elasticsearch, run:

```bash
elasticsearch-plugin install file:///$(pwd)/build/distributions/ltr-query-0.1.1-es5.3.0.zip
```

I have included a convenience scripts called `installPlugin.sh` in this directory that will go ahead and run these 
commands for you.

## Training A Model

To train a model, this project uses RankLib. This project contains a pipeline that will perform feature extraction, 
model training, re-ranking and evaluation. To run the pipeline, use: 
 
```bash
./trainltr
```

The top of the file contains variables that may be configured to change, for instance, elasticsearch settings.
 
Additionally, the pipeline comprises the following python scripts:

 - `ltrfeatures.py`: automatically extract features and produce RankLib training data.
 - `uploadscript.py`: facilitate the uploading of RankLib models into the ltr elasticsearch plugin
 - `search.py`: compare the baseline similarity scores against the learn to rank similarity function
 
## Features

Features are constructed as subclasses of `AbstractFeature` in the `features` module. See the 
[features readme](features/README.md) to explore how to extend and modify the features.
