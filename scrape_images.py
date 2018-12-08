import pandas as pd
from google_images_download import google_images_download   
import os
import re

def Scrape_from_file(keyword_file,Images_per_category=1000,Download_Folder="Downloads",delay=0.05,Image_Size="medium"):
    #Get Current Working Directory
    cwd = os.getcwd()

    data_rows=keyword_file.count()[0]

    for i in range(keyword_file.count()[0]):
        Path_List={}
        overall_path={}

        keyword=str(keyword_file.Keywords[i])
        prefix=str(keyword_file.Prefixes[i])
        
        #Adding 4 as a buffer to compensate for errors while downloading images 
        image_per_iter=int(Images_per_category/keyword_file.Total_Items[i])+4

        response = google_images_download.googleimagesdownload()   #class initialization

        prefix_keyword="-"+keyword+"-"

        arguments = {"keywords":keyword,"limit":image_per_iter,"prefix_keywords":prefix,
                            "prefix":prefix_keyword,"print_urls":False,"delay":delay,
                            "chromedriver": "chromedriver","no_numbering":"no_numbering",
                            "output_directory":str(cwd+"/"+Download_Folder+"/"+keyword),"format":"jpg",
                            "no_directory":"no_directory","size":Image_Size}   #creating list of arguments
        paths = response.download(arguments)   #passing the arguments to the function
        Path_List.update(paths)  


#Provide the path of the csv file containing the keywords
keyword_file=pd.read_csv("keywords.csv")


Scrape_from_file(keyword_file,Images_per_category=50,Download_Folder="data",Image_Size="medium",delay=0.05)
