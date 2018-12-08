#!/usr/bin/env python
#  coding: utf-8

###### Loading the images from the input directory to MongoDB ######

from pyspark.ml.image import ImageSchema
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.window import Window

sparkTrain = SparkSession \
    .builder \
    .appName("ElecNET") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/ElecNet.ImgColl") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/ElecNet.ImgColl") \
    .getOrCreate()

#Reading images from the input directory
df = ImageSchema.readImages('data', recursive=True, dropImageFailures=True)
paths = df.select(df['image']['origin'].alias('filename'), df['image']['data'].alias('image_bytes'), df['image']['width'].alias('width'), df['image']['height'].alias('height'))
split_col_filename = functions.split(paths['filename'], ':')
split_col_label = functions.split(paths['filename'], '-')
paths = paths.withColumn('category', split_col_label.getItem(1))
paths = paths.withColumn('filepath', split_col_filename.getItem(1))
paths = paths.select(paths['filepath'], paths['category'], paths['image_bytes'], paths['width'], paths['height'])

#Creating train and test sets
splits = paths.randomSplit(weights = [0.8, 0.2])
train = splits[0]
test = splits[1]
train = train.withColumn('flag',functions.lit('TR').cast(types.StringType()))
test = test.withColumn('flag',functions.lit('TE').cast(types.StringType()))
paths = train.unionAll(test)

#Creating a list of categories
categories=paths.select("category").distinct().collect()
categories=[list(i)[0] for i in categories]

#Loading the train and test sets into MongoDB
for i,category in enumerate(categories):
    paths_subset=paths.filter((paths["category"]==category))
    if(i==0):
        paths_subset.write.format("com.mongodb.spark.sql.DefaultSource").options(parallelism=8).mode("overwrite").save()
        print("Loaded:",category)
    else:
        paths_subset.write.format("com.mongodb.spark.sql.DefaultSource").options(parallelism=8).mode("append").save()
        print("Loaded:",category)