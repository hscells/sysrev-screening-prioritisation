# Features

The `ltrfeatures.py` script will use this module to extract features from elasticsearch. The base class is located in
`features.py`. Each feature must be a subclass of this abstract class by implementing the `calc()` function. The class
provides the following fields for calculations:

 - `self.statistics`: Elasticsearch term vector api response
 - `self.field`: A field from the *query*
 - `self.query`: The elasticsearch query
 - `self.query_vocabulary`: The vocabulary of the query, by each elasticsearch field
 - `self.field_statistics`: The term vector "field" statistics for the current field
 - `self.term_statistics`: The term vector "term" statistics for the current field

For an example, have a look a the `idf.py` script in this folder.