import pandas as pd
from dotenv import load_dotenv
import numpy as np
from pandas import Timestamp

import os
import json
import requests
from datetime import datetime, timezone
import zoneinfo
import warnings
import glob
import difflib
import time


import db_functions 
from importlib import reload
reload(db_functions)
 
from db_functions import fetch_data, types_fix, bronze_builds, bronze_jobs_logs, bronze_agents, calculate_wait_time, reconstruct_log, fetch_logs_for_df, fetch_job_log


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
warnings.filterwarnings('ignore')

load_dotenv()

BUILDKITE_API_TOKEN = os.getenv('BUILDKITE_API_TOKEN')
GMAIL_USERNAME = os.getenv('GMAIL_USERNAME')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
ORGANIZATION_SLUG='vllm'
PIPELINE_SLUG = 'ci'#'ci-aws'
LAST_24_HOURS = (datetime.utcnow() - pd.Timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M')
WAITING_TIME_ALERT_THR = 10800 # 3 hours
AGENT_FAILED_BUILDS_THR = 3 # agents declaired unhealthy if they have failed jobs from >=3 unique builds
RECIPIENTS = ['hissu.hyvarinen@amd.com', 'olga.miroshnichenko@amd.com', 'alexei.ivanov@amd.com']
PATH_TO_LOGS = '/mnt/home/buildkite_logs/'
PATH_TO_TABLES = '/mnt/home/vllm_fresh/.buildkite_monitor/db_demo/with_logs_table/'

LAST_2_HOURS = (datetime.utcnow() - pd.Timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M')

builds_bronze_path = PATH_TO_TABLES + 'bronze_tables/builds.parquet'
jobs_bronze_path = PATH_TO_TABLES + 'bronze_tables/jobs.parquet'
agents_bronze_path = PATH_TO_TABLES + 'bronze_tables/agents.parquet'
job_logs_bronze_path = PATH_TO_TABLES + 'bronze_tables/job_logs.parquet'
job_logs_silver_path = PATH_TO_TABLES + 'silver_tables/job_logs.parquet'

raw_data_dir = 'raw_data/'
 

# silver layer
jobs_silver_path = PATH_TO_TABLES + 'silver_tables/jobs.parquet'
builds_silver_path = PATH_TO_TABLES + 'silver_tables/builds.parquet'


#df = pd.read_parquet(job_logs_bronze_path)

def check_parquet_files(raw_data_dir, job_logs_bronze_path):
    # Get all parquet files in the raw_data directory
    parquet_files = glob.glob(os.path.join(raw_data_dir, 'logurls_*.parquet'))
    
    # Extract timestamps from filenames
    file_timestamps = [os.path.basename(f).split('_')[1].replace('.parquet', '').replace('T', ' ') for f in parquet_files]
    
    # Read the job_logs_bronze table
    job_logs_bronze = pd.read_parquet(job_logs_bronze_path)
    
    # Convert the timestamp column to string format
    job_logs_timestamps = job_logs_bronze['timestamp'].astype('str')
    
    # Check if all file timestamps are present in the job_logs_bronze table
    missing_timestamps = [ts for ts in file_timestamps if ts not in job_logs_timestamps.values]
    
    if missing_timestamps:
        print("Missing timestamps in job_logs_bronze table:", missing_timestamps)
    else:
        print("All parquet files are accounted for in the job_logs_bronze table.")
    return missing_timestamps  

def find_new_jobs(df, bronze):
    print('bronze file size', bronze.shape)
    merged_job_logs = pd.merge(bronze, df, on=['id', 'state', 'log_url'], how='right', indicator=True)
    print('merged_job_logs: ',merged_job_logs.shape)
    #display(merged_job_logs.head())
    df = merged_job_logs[merged_job_logs['_merge'] == 'right_only']
    print(f'jobs in new raw file that are not in bronze: {df.shape}')
    df = df[['timestamp_y', 'id', 'state', 'log_url']].rename(columns={'timestamp_y': 'timestamp'})
    #display(df.head())  
    return df  

def download_logs_for_df(df, token, org_slug, pipe_slug):
    start_time = time.time()
    for idx in df.id.unique():
        if df[df.id==idx].log_url.notna().values[0]:
            log, rate_limit_remaining, rate_limit_reset = fetch_job_log(df[df.id==idx].log_url.values[0], BUILDKITE_API_TOKEN, ORGANIZATION_SLUG, PIPELINE_SLUG)
            df.loc[df.id==idx, 'log'] = log
            rate_limit = int(rate_limit_remaining)
            #print(rate_limit)
            if rate_limit < 30:
                print(f'rate_limit is low = {rate_limit}, sleeping for {rate_limit_reset} seconds')
                time.sleep(int(rate_limit_reset))

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return df

bronze_job_logs_exists = os.path.exists(job_logs_bronze_path)

if not bronze_job_logs_exists:
    print("Bronze job logs table doesn't exist")
    parquet_files = glob.glob(os.path.join(raw_data_dir, 'logurls*.parquet'))
    parquet_files.sort(key=os.path.getmtime, reverse=False)
    #bronze_not_empty = False

else:
    print('bronze table with job logs exists')
    missing_ts = check_parquet_files(raw_data_dir, job_logs_bronze_path)
    if missing_ts:        
        parquet_files = [raw_data_dir + f"/logurls_{ts}.parquet".replace(' ', 'T') for ts in missing_ts.sort_values(by='timestamp')]
        print("the following parquet files haven't been read yet:", parquet_files) 
    else:
        print('all parquet files have been read already, exiting')
        raise SystemExit

    



for logurl_parq in parquet_files:
    print(logurl_parq)
    df = pd.read_parquet(logurl_parq)
    print('parquet file size', df.shape)
    df['log'] = None

    if bronze_job_logs_exists:
        print('bronze not empty')
        bronze = pd.read_parquet(job_logs_bronze_path)

        df = find_new_jobs(df, bronze)

    df = download_logs_for_df(df, BUILDKITE_API_TOKEN, ORGANIZATION_SLUG, PIPELINE_SLUG)        
    
    # makes sense to save only jobs that have not null log, but it then will save these jobs in raw since it is not yet in bronze, if logurl 
    # is null then it will be filtered out in logs_fetching, but if logurl is not null, then log will be fetched.
    # as first iteration I probably better save everything even if logurl is None, will think about it later.

    print(f'concatinating df {df.shape} to bronze parquet')
    if bronze_job_logs_exists:
        bronze = pd.concat([bronze, df], axis=0).sort_values(by='timestamp')
        bronze.to_parquet(job_logs_bronze_path)
    else:    
        df.to_parquet(job_logs_bronze_path) 
        bronze_job_logs_exists = True
    
    #display(df.head())
