# Rn

import pandas as pd
import ast
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Step 1: Read the CSV file using pandas
keywords_df = pd.read_csv("cleaned_classification_keywords.csv")

# Step 2: Convert the pandas DataFrame to a dictionary for keyword lookup
keywords_dict = {}
for index, row in keywords_df.iterrows():
    keywords_str = str(row['KEYWORDS'])
    try:
        keywords_list = ast.literal_eval(keywords_str)
        for keyword in keywords_list:
            keywords_dict[keyword.lower()] = (row['CATEGORY_LEVEL1'], row['CATEGORY_LEVEL2'])
    except (ValueError, SyntaxError) as e:
        print("Error in row {}: {}".format(index, row))
        print("Error: {}".format(e))
        print("Skipping this row...\n")

# Step 3: Initialize a Spark session
spark = SparkSession.builder.appName("Classification").getOrCreate()

# Function to set category levels
def set_category_levels(remitter_name, base_txn_text, benef_name):
    remitter_name = '' if remitter_name is None else remitter_name.lower()
    base_txn_text = '' if base_txn_text is None else base_txn_text.lower()
    benef_name = '' if benef_name is None else benef_name.lower()

    if all(x == '' for x in [remitter_name, base_txn_text, benef_name]):
        return ('MIS TRANSACTION', 'MIS TRANSACTION')
    elif remitter_name != '' and benef_name != '' and \
         any(word in benef_name.split() for word in remitter_name.split()):
        if any(keyword in base_txn_text for keyword in ['ola', 'uber', 'rent', 'purchase', 'taxi']):
            return ('SELF TRANSFER', 'SHOPPING/RENT/CAB RENT')
        else:
            return ('SELF TRANSFER', 'PERSONAL TRANSFER')
    elif remitter_name == benef_name:
        return ('SELF TRANSFER', 'PERSONAL TRANSFER')
    elif 'gifts' in base_txn_text or 'purchase' in base_txn_text:
        return ('GIFT', 'GIFTS/PURCHASES')
    elif 'credit card' in base_txn_text or 'credit card bill' in base_txn_text or \
         'credit card' in benef_name or 'credit card bill' in benef_name:
        return ('BILLS', 'CREDIT CARD BILLS/BILL PAYMENTS')
    elif 'taxi' in base_txn_text or 'uber' in base_txn_text or 'ola' in base_txn_text:
        return ('CAB PAYMENT', 'CAB RENTAL')
    elif 'shopping' in base_txn_text or 'online shopping' in base_txn_text or 'amazon' in base_txn_text or 'flipkart' in base_txn_text:
        return ('SHOPPING', 'SHOPPING/LIFESTYLE')
    else:
        # New logic using keywords_dict
        for keyword in keywords_dict:
            if keyword in remitter_name or keyword in base_txn_text or keyword in benef_name:
                return keywords_dict[keyword]
                
    return ('OTHER TRANSFER', 'OTHER')

# Correcting the returnType to be StructType
schema = StructType([
    StructField("category_level1", StringType(), False),
    StructField("category_level2", StringType(), False)
])

# Register UDF for category levels
category_udf = udf(set_category_levels, schema)

# UDF for customer type classification based on txn_amt
def classify_cust_type(txn_amt):
    print("Classifying txn_amt:", txn_amt)  # Debugging print statement
    if txn_amt > 1000000:  # 10 lac is 1000000 in numerical form
        return 'corp_transaction'
    else:
        return 'individual'

cust_type_udf = udf(classify_cust_type, StringType())

# Apply UDFs to DataFrame
#df = df.withColumn('cust_type', cust_type_udf(df['txn_amt']))
# Ensure txn_amt is of numeric type
df = df.withColumn("txn_amt", df["txn_amt"].cast(DoubleType()))

# Apply UDFs to DataFrame
df = df.withColumn('cust_type', cust_type_udf(df['txn_amt']))
df = df.withColumn('categories', category_udf('remitter_name', 'base_txn_text', 'benef_name'))
df = df.withColumn('category_level1', col('categories').getItem('category_level1'))
df = df.withColumn('category_level2', col('categories').getItem('category_level2'))
df = df.drop('categories')
